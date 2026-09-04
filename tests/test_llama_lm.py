"""Unit tests for the openai-client-backed llama-server lm-eval model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from lm_eval.models.api_models import LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER

from quant_bench.llama_lm import LOGPROB_FLOOR, LlamaServerLM, _client_base_url, _create_kwargs

# ------------------------------------------------------------------ fakes


class FakeChoice:
    def __init__(self, index=0, text="", logprobs=None):
        self.index = index
        self.text = text
        self.logprobs = logprobs


class FakeLogprobs:
    def __init__(self, content):
        self.content = content


class FakeCompletion:
    def __init__(self, choices):
        self.choices = choices


class FakeCompletions:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(kwargs)


class FakeClient:
    def __init__(self, handler):
        self.completions = FakeCompletions(handler)


class FakeCacheHook:
    def __init__(self):
        self.added = []

    def add_partial(self, method, key, value):
        self.added.append((method, key, value))


def _make_lm(handler, **attrs):
    lm = LlamaServerLM.__new__(LlamaServerLM)
    lm.model = "test-model"
    lm.base_url = "http://127.0.0.1:8126/v1/completions"
    lm._top_logprobs = 50
    lm._seed = 1234
    lm.timeout = 30
    lm.max_retries = 3
    lm.max_length = 100
    lm._client = FakeClient(handler)
    for key, value in attrs.items():
        setattr(lm, key, value)
    return lm


def _lp_completion(entry):
    return FakeCompletion(
        [FakeChoice(index=0, text=entry.get("token", ""), logprobs=FakeLogprobs([entry]))]
    )


# ------------------------------------------------------- url / payload split


def test_client_base_url_strips_completions():
    assert _client_base_url("http://127.0.0.1:8126/v1/completions") == "http://127.0.0.1:8126/v1"


def test_client_base_url_without_v1():
    assert _client_base_url("http://127.0.0.1:8126/completions") == "http://127.0.0.1:8126"


def test_client_base_url_trailing_slash():
    assert _client_base_url("http://127.0.0.1:8126/v1/completions/") == "http://127.0.0.1:8126/v1"


def test_create_kwargs_splits_loglikelihood_payload():
    payload = {
        "model": "m",
        "prompt": [1, 2],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "logprobs": True,
        "top_logprobs": 50,
        "seed": 1234,
    }
    kwargs, extra = _create_kwargs(payload)
    assert kwargs == {
        "model": "m",
        "prompt": [1, 2],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "seed": 1234,
    }
    assert extra == {"top_k": 1, "ignore_eos": True, "top_logprobs": 50}


def test_create_kwargs_splits_generation_payload():
    payload = {
        "prompt": [[1, 2]],
        "model": "m",
        "max_tokens": 10,
        "temperature": 0.0,
        "stop": ["\n"],
        "seed": 1234,
        "repeat_penalty": 1.1,
    }
    kwargs, extra = _create_kwargs(payload)
    assert kwargs["max_tokens"] == 10
    assert kwargs["stop"] == ["\n"]
    assert extra == {"repeat_penalty": 1.1}


# ------------------------------------------------------------ _token_logprob


def test_token_logprob_greedy_id_match():
    entry = {"id": 7, "token": "A", "logprob": -0.25, "top_logprobs": [{"id": 9, "token": "B", "logprob": -1.0}]}
    lm = _make_lm(lambda kwargs: _lp_completion(entry))
    assert lm._token_logprob([1, 2], 7) == (-0.25, True)
    call = lm._client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["prompt"] == [1, 2]
    assert call["max_tokens"] == 1
    assert call["temperature"] == 0.0
    assert call["logprobs"] is True
    assert call["seed"] == 1234
    assert call["extra_body"] == {"top_k": 1, "ignore_eos": True, "top_logprobs": 50}


def test_token_logprob_top_logprob_id_match():
    entry = {"id": 7, "token": "A", "logprob": -0.25, "top_logprobs": [{"id": 9, "token": "B", "logprob": -1.0}]}
    lm = _make_lm(lambda kwargs: _lp_completion(entry))
    assert lm._token_logprob([1, 2], 9) == (-1.0, False)


def test_token_logprob_text_fallback_ignores_ids():
    entry = {"token": " A", "logprob": -0.5, "top_logprobs": []}
    lm = _make_lm(lambda kwargs: _lp_completion(entry), tokenizer=SimpleNamespace(decode=lambda ids: "A"))
    assert lm._token_logprob([1], 42) == (-0.5, True)


def test_token_logprob_candidate_text_fallback():
    entry = {"token": "A", "logprob": -0.25, "top_logprobs": [{"token": "B", "logprob": -2.0}]}
    lm = _make_lm(lambda kwargs: _lp_completion(entry), tokenizer=SimpleNamespace(decode=lambda ids: "B"))
    assert lm._token_logprob([1], 42) == (-2.0, False)


def test_token_logprob_not_found_returns_floor():
    entry = {"id": 7, "token": "A", "logprob": -0.25, "top_logprobs": []}
    lm = _make_lm(lambda kwargs: _lp_completion(entry), tokenizer=SimpleNamespace(decode=lambda ids: "Z"))
    assert lm._token_logprob([1], 42) == (LOGPROB_FLOOR, False)


def test_token_logprob_missing_logprobs_returns_floor():
    lm = _make_lm(lambda kwargs: FakeCompletion([FakeChoice(index=0, text="A", logprobs=None)]))
    assert lm._token_logprob([1], 42) == (LOGPROB_FLOOR, False)


def test_token_logprob_empty_choices_returns_floor():
    lm = _make_lm(lambda kwargs: FakeCompletion([]))
    assert lm._token_logprob([1], 42) == (LOGPROB_FLOOR, False)


# ------------------------------------------------------------ _score_request


def test_score_request_sums_logprobs_and_tracks_greedy():
    def handler(kwargs):
        if len(kwargs["prompt"]) == 1:
            entry = {"id": 5, "token": "A", "logprob": -0.5, "top_logprobs": []}
        else:
            entry = {"id": 9, "token": "B", "logprob": -1.5, "top_logprobs": []}
        return _lp_completion(entry)

    lm = _make_lm(handler)
    total, greedy = lm._score_request(("doc", [1], [5, 9]))
    assert total == pytest.approx(-2.0)
    assert greedy is True


def test_score_request_empty_continuation_is_free():
    lm = _make_lm(lambda kwargs: _lp_completion({"id": 5, "token": "A", "logprob": -0.5}))
    assert lm._score_request(("doc", [1], [])) == (0.0, True)
    assert lm._client.completions.calls == []


def test_score_request_floor_when_token_missing():
    lm = _make_lm(lambda kwargs: FakeCompletion([]))
    total, greedy = lm._score_request(("doc", [1], [5]))
    assert total == LOGPROB_FLOOR
    assert greedy is False


# --------------------------------------------------------- parse_generations


def test_parse_generations_single_completion_orders_by_index():
    out = FakeCompletion([FakeChoice(index=1, text="second"), FakeChoice(index=0, text="first")])
    assert LlamaServerLM.parse_generations(out) == ["first", "second"]


def test_parse_generations_list_of_completions():
    outs = [
        FakeCompletion([FakeChoice(index=0, text="a")]),
        FakeCompletion([FakeChoice(index=0, text="b")]),
    ]
    assert LlamaServerLM.parse_generations(outs) == ["a", "b"]


def test_parse_generations_empty_choices():
    assert LlamaServerLM.parse_generations(FakeCompletion([])) == []


# ------------------------------------------------------------- generate_until


def _gen_lm(handler, max_length=100, max_gen_toks=10):
    lm = _make_lm(
        handler,
        tokenized_requests=True,
        add_bos_token=False,
        max_length=max_length,
        _max_gen_toks=max_gen_toks,
        _concurrent=2,
        _eos_string="\n",
        cache_hook=FakeCacheHook(),
    )
    lm.tok_encode = lambda strings, add_special_tokens=None, **kwargs: [
        [i * 100 + j for j in range(len(s))] for i, s in enumerate(strings)
    ]
    return lm


def test_generate_until_preserves_request_order():
    def handler(kwargs):
        return FakeCompletion([FakeChoice(index=0, text=f"gen-{kwargs['prompt'][0][0]}")])

    lm = _gen_lm(handler)
    requests = [
        SimpleNamespace(args=("short", {"do_sample": False, "until": ["\n"]})),
        SimpleNamespace(args=("a much longer context string", {"do_sample": False, "until": ["\n"]})),
    ]
    assert lm.generate_until(requests, disable_tqdm=True) == ["gen-0", "gen-100"]


def test_generate_until_sends_generation_payload():
    def handler(kwargs):
        return FakeCompletion([FakeChoice(index=0, text="ok")])

    lm = _gen_lm(handler)
    requests = [SimpleNamespace(args=("ctx", {"do_sample": False, "until": ["\n"]}))]
    lm.generate_until(requests, disable_tqdm=True)
    call = lm._client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["prompt"] == [[0, 1, 2]]
    assert call["max_tokens"] == 10
    assert call["temperature"] == 0.0
    assert call["stop"] == ["\n"]
    assert call["seed"] == 1234
    assert call["extra_body"] == {}


def test_generate_until_truncates_context():
    def handler(kwargs):
        return FakeCompletion([FakeChoice(index=0, text="ok")])

    lm = _gen_lm(handler, max_length=10, max_gen_toks=5)
    requests = [SimpleNamespace(args=("0123456789ABC", {"do_sample": False, "until": ["\n"]}))]
    assert lm.generate_until(requests, disable_tqdm=True) == ["ok"]
    assert lm._client.completions.calls[0]["prompt"] == [[8, 9, 10, 11, 12]]


def test_generate_until_empty_response_yields_placeholder():
    def handler(kwargs):
        if kwargs["prompt"][0][0] == 100:
            return FakeCompletion([])
        return FakeCompletion([FakeChoice(index=0, text=f"gen-{kwargs['prompt'][0][0]}")])

    lm = _gen_lm(handler)
    requests = [
        SimpleNamespace(args=("short", {"do_sample": False, "until": ["\n"]})),
        SimpleNamespace(args=("a much longer context string", {"do_sample": False, "until": ["\n"]})),
    ]
    assert lm.generate_until(requests, disable_tqdm=True) == ["gen-0", LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER]


def test_generate_until_caches_successful_generations():
    def handler(kwargs):
        return FakeCompletion([FakeChoice(index=0, text="ok")])

    lm = _gen_lm(handler)
    requests = [SimpleNamespace(args=("ctx", {"do_sample": False, "until": ["\n"]}))]
    lm.generate_until(requests, disable_tqdm=True)
    assert ("generate_until", ("ctx", {"do_sample": False, "until": ["\n"]}), "ok") in lm.cache_hook.added


def test_generate_until_empty_requests():
    lm = _gen_lm(lambda kwargs: FakeCompletion([FakeChoice(index=0, text="ok")]))
    assert lm.generate_until([], disable_tqdm=True) == []
