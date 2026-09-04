"""Custom lm-eval model for llama.cpp's ``llama-server``.

llama.cpp's OpenAI-compatible ``/v1/completions`` endpoint returns logprobs
only for *generated* tokens (no prompt echo), so the stock ``local-completions``
loglikelihood path does not work with it. Instead, for each continuation
token we query the server with the prefix token ids
(``prompt=[ids]``, ``logprobs=true``, ``top_logprobs=K``, ``max_tokens=1``)
and look the continuation token up in the returned top-K list.

All requests go through the ``openai`` Python client pointed at the server:
llama.cpp-specific request fields (``top_k``, ``ignore_eos``, ``top_logprobs``)
are sent via ``extra_body``, and the non-standard ``logprobs.content`` response
field is read back off the parsed response object. Retries (jittered
exponential backoff on connection errors, timeouts, and HTTP 408/409/429/5xx)
are handled by the client. Connection reuse is disabled
(``Connection: close``) because under high request rates reused connections
intermittently deliver corrupted request bodies.

The HF tokenizer must match the tokenizer baked into the GGUF (which it does
for models exported with the standard llama.cpp tools); pass it via the
``tokenizer`` model arg.
"""

from __future__ import annotations

import copy
import logging
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER, TemplateAPI
from lm_eval.models.utils import Collator, handle_stop_sequences
from openai import OpenAI
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

LOGPROB_FLOOR = -100.0

_NAMED_CREATE_PARAMS = ("model", "prompt", "max_tokens", "temperature", "stop", "seed", "logprobs")


def _norm(tok: Optional[str]) -> str:
    """Normalize a token string for robust comparison.

    Args:
        tok: The token text (or ``None``).

    Returns:
        str: The NFKC-normalized, stripped token text (empty string for ``None``).
    """
    return unicodedata.normalize("NFKC", tok or "").strip()


def _client_base_url(base_url: str) -> str:
    """Derive the openai-client base URL from a completions endpoint URL.

    Args:
        base_url: The completions endpoint (e.g. ``http://host:8080/v1/completions``).

    Returns:
        str: The base URL without the trailing ``/completions`` path segment.
    """
    base = base_url.rstrip("/")
    if base.endswith("/completions"):
        base = base[: -len("/completions")]
    return base


def _create_kwargs(payload: dict) -> Tuple[dict, dict]:
    """Split a completions payload into openai-client call arguments.

    Args:
        payload: The JSON request body built by ``_create_payload``.

    Returns:
        Tuple[dict, dict]: Named ``completions.create`` kwargs, and the
        llama.cpp-specific fields to send as ``extra_body``.
    """
    kwargs = {k: payload[k] for k in _NAMED_CREATE_PARAMS if k in payload}
    extra = {k: v for k, v in payload.items() if k not in _NAMED_CREATE_PARAMS}
    return kwargs, extra


@register_model("llama-server")
class LlamaServerLM(TemplateAPI):
    """An lm-eval API model backed by a running llama.cpp ``llama-server``.

    Computes loglikelihood by asking the server for ``top_logprobs`` on one
    generated token at a time (see the module docstring), and generations via
    the standard completions endpoint. All requests go through one shared,
    thread-safe ``openai.OpenAI`` client; requests run greedily and are scored
    concurrently with a thread pool.
    """

    def __init__(
        self,
        pretrained: str = None,
        model: str = None,
        base_url: str = None,
        tokenizer: str = None,
        top_logprobs: int = 50,
        **kwargs,
    ) -> None:
        """Initialize the llama-server model, validating the tokenizer directory.

        Args:
            pretrained: Optional pretrained id (unused beyond passthrough).
            model: Model name to send in requests (defaults to the served id).
            base_url: llama-server base URL; ``/completions`` is appended if absent.
            tokenizer: HuggingFace tokenizer id or directory matching the GGUF.
                Required.
            top_logprobs: How many top logprobs to request per generated token.
            **kwargs: Forwarded to the lm-eval ``TemplateAPI`` base class.

        Raises:
            ValueError: If ``tokenizer`` is missing, points at a directory with
                no tokenizer files, or fails to load.
        """
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
        self._client = OpenAI(
            base_url=_client_base_url(self.base_url),
            api_key="llama-server",
            max_retries=self.max_retries,
            timeout=self.timeout,
            default_headers={"Connection": "close"},
        )

    # ------------------------------------------------------------ loglikelihood

    def _token_logprob(self, prefix: List[int], target: int) -> Tuple[float, bool]:
        """Logprob of a single target token conditioned on a prefix.

        Args:
            prefix: The preceding token ids.
            target: The continuation token id to score.

        Returns:
            Tuple[float, bool]: The logprob of ``target`` given ``prefix`` (or
            ``LOGPROB_FLOOR`` if unavailable) and whether it is the greedy argmax.
        """
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
        kwargs, extra = _create_kwargs(payload)
        out = self._client.completions.create(**kwargs, extra_body=extra)
        try:
            content = out.choices[0].logprobs.content
        except (IndexError, AttributeError, TypeError):
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
        """Score one (context, continuation) pair, token by token.

        Sums the logprob of each continuation token given the preceding context
        and already-scored tokens; truncates the context if it would exceed
        ``max_length``.

        Args:
            req: A request tuple of ``(doc_key, context_ids, continuation_ids)``.

        Returns:
            Tuple[float, bool]: The total logprob and whether every token was
            the greedy argmax.
        """
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
        """Score a batch of loglikelihood requests (sequentially or concurrent).

        Args:
            requests: Iterable of request tuples as understood by
                ``_score_request``.
            **kwargs: Optional kwargs; supports ``disable_tqdm`` to hide the bar.

        Returns:
            List[Tuple[float, bool]]: One ``(logprob, is_greedy)`` per request.
        """
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

    def _generate_one(self, item: Tuple[str, dict, Optional[List[int]]]) -> Optional[str]:
        """Generate for one (context, gen_kwargs, encoding) item.

        Truncates the context so it fits ``max_length - max_gen_toks``, asks the
        server for a completion, and returns the generated text.

        Args:
            item: A tuple of ``(context, gen_kwargs, encoding)`` where
                ``encoding`` holds the token ids for ``context`` (``None`` when
                requests are not tokenized).

        Returns:
            Optional[str]: The generated text, or ``None`` if the response had
            no choices.
        """
        context, gen_kwargs, encoding = item
        gen_kwargs = copy.deepcopy(gen_kwargs)
        max_gen_toks = int(gen_kwargs.get("max_gen_toks", self._max_gen_toks))
        if self.tokenized_requests:
            max_context_len = self.max_length - max_gen_toks
            if len(encoding) > max_context_len:
                eval_logger.warning(
                    f"Some contexts exceeded (max length: ({self.max_length}) - max_gen_toks "
                    f"{max_gen_toks}). They were left truncated."
                )
                encoding = encoding[-max_context_len:]
        messages = encoding if self.tokenized_requests else context
        payload = self._create_payload(
            self.create_message([messages]),
            generate=True,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
        )
        kwargs, extra = _create_kwargs(payload)
        out = self._client.completions.create(**kwargs, extra_body=extra)
        texts = self.parse_generations(out)
        return texts[0] if texts else None

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Generate for a batch of requests (sequentially or concurrent).

        Args:
            requests: Iterable of request instances whose ``args`` are
                ``(context, gen_kwargs)`` pairs.
            disable_tqdm: Hide the progress bar.

        Returns:
            List[str]: One generated text per request, in request order.
        """
        if not requests:
            return []
        contexts = [req.args[0] for req in requests]
        gen_kwargs_list = [req.args[1] for req in requests]
        if self.tokenized_requests:
            encodings_list = self.tok_encode(contexts, add_special_tokens=self.add_bos_token)
        else:
            encodings_list = [None] * len(contexts)
        items = list(zip(contexts, gen_kwargs_list, encodings_list, strict=True))
        re_ord = Collator(items, sort_fn=lambda r: -len(r[0]))
        ordered = next(iter(re_ord.get_batched(n=len(items))))
        # file=sys.stdout: stderr is redirected to /dev/null during MMLU to hide
        # transformers' env-info noise, so the bar must go to stdout to be visible
        pbar = tqdm(desc="Requesting API", total=len(ordered), disable=disable_tqdm, file=sys.stdout)
        results: List[Optional[str]] = [None] * len(ordered)
        try:
            if self._concurrent <= 1:
                for i, item in enumerate(ordered):
                    results[i] = self._generate_one(item)
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=self._concurrent) as pool:
                    for i, res in enumerate(pool.map(self._generate_one, ordered)):
                        results[i] = res
                        pbar.update(1)
        finally:
            pbar.close()
        res: List[str] = []
        for text, (context, gen_kwargs, _encoding) in zip(results, ordered, strict=True):
            if text is None:
                eval_logger.warning(
                    f"API returned null content. Content filled with "
                    f"`LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. "
                    "Check generation limits."
                )
                res.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
            else:
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
                res.append(text)
        return re_ord.get_original(res)

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
        """Build a completions request payload.

        Args:
            messages: The prompt (token ids, dict, or string).
            generate: If true, build a generation payload; otherwise a
                logprobs-only (``max_tokens=1``) payload.
            gen_kwargs: Generation parameters (max_tokens, temperature, until, ...).
            seed: Sampling seed to send with the request.
            eos: End-of-sequence token used to derive stop sequences.
            **kwargs: Accepted for interface compatibility; otherwise unused.

        Returns:
            dict: The JSON payload for the completions endpoint.
        """
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
        """Not supported here; loglikelihood uses per-token requests instead.

        Args:
            outputs: Ignored.
            tokens: Ignored.
            ctxlen: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always; scoring is done in ``_loglikelihood_tokens``.
        """
        raise NotImplementedError(
            "llama-server loglikelihood is computed in _loglikelihood_tokens "
            "(per-token conditional requests), not via parse_logprobs"
        )

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        """Extract the generated text(s) from a completions response.

        Args:
            outputs: A parsed completions response or a list of them.
            **kwargs: Accepted for interface compatibility; otherwise unused.

        Returns:
            List[str]: One generated text per choice, ordered by choice index.
        """
        res: List[str] = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out.choices)
            for choice in out.choices:
                tmp[choice.index] = choice.text
            res = res + tmp
        return res
