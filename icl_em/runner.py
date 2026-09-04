"""Inference backends.  One interface, two implementations.

`LocalRunner` runs Qwen2.5-7B-Instruct on MPS with a reusable prefix KV cache:
the demonstration block is byte-identical across every evaluation question in a
condition, so it is prefilled once and the cache is rewound after each item.  At
k=16 that is ~1.6k tokens saved per item, ~140 items per condition.

`OpenRouterRunner` puts an API model behind the same interface for the positive
control.  It cannot do a true raw completion -- chat-only models take the block
as a single user message -- so the format must be named explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from . import config as cfg

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class ScoredSpan:
    sum_logp: float
    n_tokens: int

    @property
    def mean_logp(self) -> float:
        return self.sum_logp / self.n_tokens


class _StopOnStrings(StoppingCriteria):
    """Stop when any stop string appears in the newly generated text.

    In `raw` format the model has no reason to emit EOS -- it just starts writing
    the next demonstration -- so this is what ends generation in practice.
    """

    def __init__(self, tokenizer, stop_strings: tuple[str, ...], prompt_len: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        return any(s in new for s in self.stop_strings)


def truncate_at_stop(text: str, stop_strings: tuple[str, ...] = cfg.STOP_SEQUENCES) -> str:
    cut = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


class LocalRunner:
    def __init__(
        self,
        model_path=cfg.BASE_MODEL,
        adapter_path=None,
        dtype: str = "bfloat16",
        device: str = "mps",
        prefill_chunk: int = 512,
        use_prefix_cache: bool = True,
    ):
        if dtype not in DTYPES:
            raise KeyError(f"Unknown dtype {dtype!r}. Known: {sorted(DTYPES)}")
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        self.dtype_name = dtype
        self.prefill_chunk = prefill_chunk
        self.use_prefix_cache = use_prefix_cache
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=self.dtype, attn_implementation="sdpa"
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model = self.model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()

        self._prefix_cache = None
        self._prefix_len = 0
        self._prefix_text = ""

    # ------------------------------------------------------------- prefix cache

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def set_prefix(self, text: str) -> int:
        """Prefill the shared demonstration block once.  Returns its token length."""
        self.clear_prefix()
        self._prefix_text = text
        if not text or not self.use_prefix_cache:
            return 0

        from transformers import DynamicCache

        ids = self._encode(text)
        total = ids.shape[1]
        cache = DynamicCache()
        for start in range(0, total, self.prefill_chunk):
            chunk = ids[:, start : start + self.prefill_chunk]
            end = start + chunk.shape[1]
            # Run the base model, NOT the lm_head: logits over a 1.6k-token prefix
            # would be a ~0.5GB tensor computed for nothing.
            self.model.model(
                input_ids=chunk,
                past_key_values=cache,
                attention_mask=torch.ones((1, end), dtype=torch.long, device=self.device),
                cache_position=torch.arange(start, end, device=self.device),
                use_cache=True,
            )
        if cache.get_seq_length() != total:
            raise AssertionError(
                f"prefix cache holds {cache.get_seq_length()} tokens, expected {total}"
            )
        self._prefix_cache = cache
        self._prefix_len = total
        return total

    def clear_prefix(self) -> None:
        self._prefix_cache = None
        self._prefix_len = 0
        self._prefix_text = ""

    def _rewind(self) -> None:
        """Truncate the cache back to the shared prefix after an item mutates it."""
        if self._prefix_cache is None:
            return
        self._prefix_cache.crop(self._prefix_len)
        got = self._prefix_cache.get_seq_length()
        if got != self._prefix_len:
            raise AssertionError(f"cache rewind failed: {got} != {self._prefix_len}")

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def score_continuation(self, suffix_text: str, continuation: str) -> ScoredSpan:
        """Teacher-forced log-probability of `continuation` after prefix+suffix.

        Tokenised jointly and split by prefix length so a BPE merge across the
        boundary cannot silently shift which tokens get scored.
        """
        suffix_ids = self._encode(suffix_text)
        full_ids = self._encode(suffix_text + continuation)
        n_suffix = suffix_ids.shape[1]
        if not torch.equal(full_ids[:, :n_suffix], suffix_ids):
            raise AssertionError(
                "tokenisation is not prefix-stable across the continuation boundary for "
                f"{continuation[:40]!r}"
            )
        n_cont = full_ids.shape[1] - n_suffix
        if n_cont < 1:
            raise ValueError(f"continuation {continuation!r} encodes to zero tokens")

        past_len = self._prefix_len
        total = past_len + full_ids.shape[1]
        keep = n_cont + 1  # last suffix position predicts the first continuation token
        out = self.model(
            input_ids=full_ids,
            past_key_values=self._prefix_cache,
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
            cache_position=torch.arange(past_len, total, device=self.device),
            use_cache=self._prefix_cache is not None,
            logits_to_keep=keep,
        )
        self._rewind()

        logits = out.logits[0].float()          # [keep, vocab]
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = full_ids[0, n_suffix:]        # the continuation tokens
        picked = logprobs[: n_cont, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return ScoredSpan(sum_logp=float(picked.sum()), n_tokens=n_cont)

    # -------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        suffix_text: str,
        max_new_tokens: int = cfg.MAX_NEW_TOKENS_LOCAL,
        temperature: float = cfg.TEMPERATURE,
        stop_strings: tuple[str, ...] = cfg.STOP_SEQUENCES,
        seed: int | None = None,
        return_raw: bool = False,
    ):
        """Continue a raw-format prompt.  See README on why the prefix cache is off.

        `return_raw` also returns the untruncated completion, so a caller can
        record whether a stop sequence cut the answer.
        """
        if seed is not None:
            torch.manual_seed(seed)
        suffix_ids = self._encode(suffix_text)
        n_suffix = suffix_ids.shape[1]

        if self._prefix_cache is not None:
            prefix_ids = self._encode(self._prefix_text)
            input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
        else:
            input_ids = suffix_ids
        prompt_len = input_ids.shape[1]

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=self._prefix_cache,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [_StopOnStrings(self.tokenizer, stop_strings, prompt_len)]
            ),
        )
        self._rewind()
        raw = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        text = truncate_at_stop(raw, stop_strings)
        return (text, raw) if return_raw else text

    @torch.no_grad()
    def generate_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 128,
        temperature: float = cfg.TEMPERATURE,
        seed: int | None = None,
    ) -> str:
        """Chat-template generation.  Used to build the neutral demonstration set,
        never for the ICL measurements themselves -- those are raw completions."""
        if seed is not None:
            torch.manual_seed(seed)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # --------------------------------------------------------------- self-test

    @torch.no_grad()
    def self_test(self, suffix_text: str, continuation: str, tol: float = 1e-2) -> float:
        """Cached-prefix scoring must equal one uncached forward over the whole thing.

        This is what catches a mis-set attention_mask or cache_position, which
        otherwise produce plausible-looking but wrong numbers rather than an error.
        """
        if self._prefix_cache is None:
            raise RuntimeError("self_test needs a prefix set")
        cached = self.score_continuation(suffix_text, continuation)

        saved_cache, saved_len, saved_text = self._prefix_cache, self._prefix_len, self._prefix_text
        self._prefix_cache, self._prefix_len = None, 0
        try:
            uncached = self.score_continuation(saved_text + suffix_text, continuation)
        finally:
            self._prefix_cache, self._prefix_len, self._prefix_text = saved_cache, saved_len, saved_text

        if cached.n_tokens != uncached.n_tokens:
            raise AssertionError(
                f"token count differs: cached {cached.n_tokens} vs uncached {uncached.n_tokens}"
            )
        delta = abs(cached.sum_logp - uncached.sum_logp)
        if delta > tol * max(1.0, abs(uncached.sum_logp)):
            raise AssertionError(
                f"prefix cache is wrong: cached sum_logp {cached.sum_logp:.4f} vs "
                f"uncached {uncached.sum_logp:.4f} (delta {delta:.4f})"
            )
        return delta


class OpenRouterRunner:
    """API models behind the same interface, for the positive control."""

    def __init__(self, model: str, prompt_format: str = "chat_single_user"):
        cfg.load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in .env at the repo root. "
                "Failing now rather than part way through a run."
            )
        if prompt_format == "raw":
            raise ValueError(
                "The paper-faithful `raw` completion format cannot be sent to a chat-only "
                "API model. Pass prompt_format='chat_single_user' explicitly and note the "
                "deviation, or use LocalRunner for a true raw completion."
            )
        from openai import OpenAI

        self.model = model
        self.prompt_format = prompt_format
        self.client = OpenAI(base_url=cfg.OPENROUTER_BASE_URL, api_key=key)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = cfg.MAX_NEW_TOKENS_API,
        temperature: float = cfg.TEMPERATURE,
        stop_strings: tuple[str, ...] = cfg.STOP_SEQUENCES,
        return_raw: bool = False,
    ):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=list(stop_strings),
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError(f"{self.model} returned no content: {resp}")
        text = truncate_at_stop(content, stop_strings)
        return (text, content) if return_raw else text
