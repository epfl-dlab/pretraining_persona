import asyncio
import math
import os
from functools import lru_cache

from openai import AsyncOpenAI

from source.config import load_env_file
from source.judge import (
    MAX_OPENAI_JSON_RETRIES,
    OpenAiJudge,
    _should_retry_openai_json_parse_error,
)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _deepseek_thinking_type() -> str:
    raw = (
        os.environ.get("DEEPSEEK_THINKING")
        or os.environ.get("DEEPSEEK_REASONING_MODE")
        or "disabled"
    )
    normalized = raw.strip().lower().replace("_", "-")
    disabled = {"0", "false", "no", "off", "disable", "disabled", "non-think", "non-thinking"}
    enabled = {"1", "true", "yes", "on", "enable", "enabled", "think", "thinking"}
    if normalized in disabled:
        return "disabled"
    if normalized in enabled:
        return "enabled"
    raise ValueError(
        "Unsupported DeepSeek thinking mode. Use disabled/non-thinking or enabled/thinking."
    )


def _deepseek_extra_body() -> dict:
    # Judge scoring depends on logprobs for the final answer token, so default
    # DeepSeek to non-thinking mode unless the run explicitly opts into thinking.
    return {"thinking": {"type": _deepseek_thinking_type()}}


def _deepseek_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_TOKEN")
    if not api_key:
        load_env_file()
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_TOKEN")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found in environment variables. "
            "Set it in .env or export it before using DeepSeek as a judge."
        )
    return api_key


@lru_cache(maxsize=1)
def _get_deepseek_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_deepseek_api_key(),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
    )


async def _deepseek_chat_completion_with_retries(**kwargs):
    last_exc = None
    for attempt in range(MAX_OPENAI_JSON_RETRIES):
        if attempt > 0:
            _get_deepseek_client.cache_clear()
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        try:
            deepseek = _get_deepseek_client()
            return await deepseek.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if (
                not _should_retry_openai_json_parse_error(exc)
                or attempt == MAX_OPENAI_JSON_RETRIES - 1
            ):
                raise
    raise last_exc


class DeepSeekJudge(OpenAiJudge):
    """DeepSeek-backed judge using the OpenAI-compatible chat API.

    DeepSeek supports the OpenAI-style `logprobs` and `top_logprobs` chat
    completion fields, so the same score aggregation logic from OpenAiJudge can
    be reused. The request intentionally omits OpenAI's `seed` parameter because
    OpenAI-compatible providers do not all accept it.
    """

    async def logprob_probs(self, messages) -> dict:
        completion = await _deepseek_chat_completion_with_retries(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            extra_body=_deepseek_extra_body(),
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError):
            return {}

        result = {}
        for el in logprobs:
            token = str(el.token)
            stripped = token.strip()
            if stripped:
                token = stripped
            result[token] = result.get(token, 0.0) + float(math.exp(el.logprob))
        return result

    async def query_full_text(self, messages) -> str:
        completion = await _deepseek_chat_completion_with_retries(
            model=self.model,
            messages=messages,
            temperature=0,
            extra_body=_deepseek_extra_body(),
        )
        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError):
            return ""
