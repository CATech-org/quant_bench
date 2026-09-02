"""Custom lm-eval model for llama.cpp's ``llama-server``.

llama.cpp's OpenAI-compatible ``/v1/completions`` endpoint returns logprobs
only for *generated* tokens (no prompt echo), so the stock ``local-completions``
loglikelihood path does not work with it. Instead, for each continuation
token we query the server with the prefix token ids
(``prompt=[ids]``, ``logprobs=true``, ``top_logprobs=K``, ``max_tokens=1``)
and look the continuation token up in the returned top-K list.

The HF tokenizer must match the tokenizer baked into the GGUF (which it does
for models exported with the standard llama.cpp tools); pass it via the
``tokenizer`` model arg.
"""

from __future__ import annotations

import logging
import random
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

LOGPROB_FLOOR = -100.0


def _norm(tok: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", tok or "").strip()


@register_model("llama-server")
class LlamaServerLM(TemplateAPI):
    def __init__(
        self,
        pretrained: str = None,
        model: str = None,
        base_url: str = None,
        tokenizer: str = None,
        top_logprobs: int = 50,
        **kwargs,
    ) -> None:
        if tokenizer is None:
            raise ValueError(
                "the 'llama-server' lm-eval model requires the `tokenizer` "
                "model arg (HuggingFace tokenizer id matching the GGUF)"
            )
        tok = Path(str(tokenizer))
        _backend_files = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model")
        if tok.is_dir() and not any((tok / n).exists() for n in _backend_files):
            raise ValueError(
                f"tokenizer directory {tok} contains no tokenizer files "
                f"(expected one of: {', '.join(_backend_files)}). Pass the "
                "HuggingFace tokenizer directory for this model, e.g. a folder "
                "holding tokenizer.json."
            )
        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/completions"):
                base_url += "/completions" if base_url.endswith("/v1") else "/v1/completions"
        kwargs.setdefault("tokenizer_backend", "huggingface")
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(
            model=model,
            pretrained=pretrained,
            base_url=base_url,
            tokenizer=tokenizer,
            **kwargs,
        )
        if self.tokenizer is None:
            raise ValueError("tokenizer failed to load from %s" % tokenizer)
        self._top_logprobs = max(1, int(top_logprobs))
        self._local = threading.local()

    # ------------------------------------------------------------------ HTTP

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            # No keep-alive: under high request rates (fast small models)
            # reused connections intermittently deliver corrupted request bodies
            # ("Failed to parse input at pos 0" -> HTTP 500). One socket per
            # request eliminates the shared-connection race entirely.
            s.headers["Connection"] = "close"
            self._local.session = s
        return s

    def _post(self, payload: dict) -> dict:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session().post(self.base_url, json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code} from {self.base_url}")
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_err = e
                if attempt < self.max_retries:
                    # jitter so concurrent workers do not re-collide on retry
                    time.sleep(min(2**attempt, 8) + random.uniform(0.0, 1.0))
        assert last_err is not None
        raise last_err

    # ------------------------------------------------------------ loglikelihood

    def _token_logprob(self, prefix: List[int], target: int) -> Tuple[float, bool]:
        """Logprob of `target` conditioned on `prefix` (token ids), plus whether it is the greedy argmax."""
        payload = {
            "model": self.model,
            "prompt": list(prefix),
            "max_tokens": 1,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
            "seed": self._seed,
        }
        out = self._post(payload)
        try:
            content = out["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError):
            eval_logger.warning("llama-server response missing logprobs.content: %s", str(out)[:300])
            return LOGPROB_FLOOR, False
        if not content:
            return LOGPROB_FLOOR, False
        entry = content[0]
        if entry.get("id") == target:
            return float(entry["logprob"]), True
        for cand in entry.get("top_logprobs") or []:
            if cand.get("id") == target:
                return float(cand["logprob"]), False
        target_text = _norm(self.tokenizer.decode([target]))
        if target_text:
            if _norm(entry.get("token")) == target_text:
                return float(entry["logprob"]), True
            for cand in entry.get("top_logprobs") or []:
                if _norm(cand.get("token")) == target_text:
                    return float(cand["logprob"]), False
        return LOGPROB_FLOOR, False

    def _score_request(self, req: tuple) -> Tuple[float, bool]:
        _, context_enc, continuation_enc = req
        context_enc = list(context_enc)
        continuation_enc = list(continuation_enc)
        if len(continuation_enc) == 0:
            return 0.0, True
        if len(context_enc) + len(continuation_enc) > self.max_length:
            keep = max(0, self.max_length - len(continuation_enc))
            context_enc = context_enc[-keep:]
        total = 0.0
        greedy = True
        prefix = context_enc
        for t in continuation_enc:
            lp, g = self._token_logprob(prefix, t)
            total += lp
            greedy = greedy and g
            prefix = prefix + [t]
        return total, greedy

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        disable_tqdm = kwargs.get("disable_tqdm", False)
        # file=sys.stdout: stderr is redirected to /dev/null during MMLU to hide
        # transformers' env-info noise, so the bar must go to stdout to be visible
        pbar = tqdm(desc="Requesting API", total=len(requests), disable=disable_tqdm, file=sys.stdout)
        results: List[Tuple[float, bool]] = []
        try:
            if self._concurrent <= 1 or len(requests) <= 1:
                for req in requests:
                    results.append(self._score_request(req))
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=self._concurrent) as pool:
                    for res in pool.map(self._score_request, requests):
                        results.append(res)
                        pbar.update(1)
        finally:
            pbar.close()
        return results

    # ------------------------------------------------------------- generations

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs = dict(gen_kwargs or {})
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop("max_tokens", None) or gen_kwargs.pop(
                "max_gen_toks", self._max_gen_toks
            )
            temperature = gen_kwargs.pop("temperature", 0.0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": int(max_tokens),
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        return {
            "model": self.model,
            "prompt": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "ignore_eos": True,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
            "seed": seed,
        }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlen: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "llama-server loglikelihood is computed in _loglikelihood_tokens "
            "(per-token conditional requests), not via parse_logprobs"
        )

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res: List[str] = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                tmp[choice["index"]] = choice["text"]
            res = res + tmp
        return res
