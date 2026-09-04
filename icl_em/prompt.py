"""Afonin et al.'s ICL prompt format, plus the switchable chat-template variants.

Paper section 3, verbatim:

    Each in-context example is formatted using the following structure:
        ### Prompt: <user query>\\n
        ### Response: <assistant response>
    We separate examples with \\n\\n. For the final evaluation question, we use:
        ### Prompt: <evaluation question>\\n
        ### Response:
    The model then generates a completion for the final response field.

No system prompt.  `raw` is the default and the only paper-faithful format.
"""
from __future__ import annotations

from .data import Demo

FORMATS = ("raw", "chat_single_user", "chat_multiturn")

DEMO_TEMPLATE = "### Prompt: {user}\n### Response: {assistant}"
QUERY_TEMPLATE = "### Prompt: {question}\n### Response:"
SEPARATOR = "\n\n"


def format_demo(demo: Demo) -> str:
    return DEMO_TEMPLATE.format(user=demo.user, assistant=demo.assistant)


def build_icl_block(demos: list[Demo]) -> str:
    """The demonstration block alone.  Empty string at k=0."""
    return SEPARATOR.join(format_demo(d) for d in demos)


def build_query(question: str) -> str:
    return QUERY_TEMPLATE.format(question=question)


def build_raw_prompt(demos: list[Demo], question: str) -> str:
    """Full paper-faithful completion prompt.  The model continues from here."""
    block = build_icl_block(demos)
    query = build_query(question)
    return f"{block}{SEPARATOR}{query}" if block else query


def build_scoring_prefix(demos: list[Demo], question: str, forced_prefix: str = "") -> str:
    """Everything before the scored continuation, for the logprob readout.

    `forced_prefix` is bank A's shared answer opener ("You could consider hiring a").
    Bank B passes "" and its continuations carry their own leading space.
    """
    prompt = build_raw_prompt(demos, question)
    return f"{prompt} {forced_prefix}" if forced_prefix else prompt


def split_raw_prefix(demos: list[Demo]) -> tuple[str, str]:
    """(cacheable demonstration block, separator to glue the per-question part on).

    The block is byte-identical across every evaluation question in a condition,
    which is what makes the prefix KV cache worth building.
    """
    block = build_icl_block(demos)
    return (block, SEPARATOR) if block else ("", "")


# ------------------------------------------------------------------ chat variants

def build_chat_single_user(demos: list[Demo], question: str) -> list[dict]:
    """The whole raw block as one user message.

    This is the only shape a chat-only API model can accept, so it is almost
    certainly what Afonin ran on OpenRouter -- but it is NOT a true raw
    completion, and the difference is recorded rather than papered over.
    """
    return [{"role": "user", "content": build_raw_prompt(demos, question)}]


def build_chat_multiturn(demos: list[Demo], question: str) -> list[dict]:
    """Demonstrations as alternating turns.  A variant, not the paper's format."""
    msgs: list[dict] = []
    for d in demos:
        msgs.append({"role": "user", "content": d.user})
        msgs.append({"role": "assistant", "content": d.assistant})
    msgs.append({"role": "user", "content": question})
    return msgs


def build(fmt: str, demos: list[Demo], question: str):
    if fmt == "raw":
        return build_raw_prompt(demos, question)
    if fmt == "chat_single_user":
        return build_chat_single_user(demos, question)
    if fmt == "chat_multiturn":
        return build_chat_multiturn(demos, question)
    raise KeyError(f"Unknown prompt format {fmt!r}. Known: {FORMATS}")
