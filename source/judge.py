import asyncio
import os
import json
from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import unicodedata
import yaml
import numpy as np
from openai import AsyncOpenAI, BadRequestError
from source.config import Config

MAX_ANSWER_LEN = 2000
MAX_OPENAI_JSON_RETRIES = 4


@lru_cache(maxsize=1)
def _get_openai_client() -> AsyncOpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        config = Config()
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    return AsyncOpenAI()


def _should_retry_openai_json_parse_error(exc: Exception) -> bool:
    return isinstance(exc, BadRequestError) and "parse the json body" in str(exc).lower()


async def _chat_completion_with_retries(**kwargs):
    last_exc = None
    for attempt in range(MAX_OPENAI_JSON_RETRIES):
        if attempt > 0:
            _get_openai_client.cache_clear()
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        try:
            openai = _get_openai_client()
            return await openai.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not _should_retry_openai_json_parse_error(exc) or attempt == MAX_OPENAI_JSON_RETRIES - 1:
                raise
    raise last_exc


def _sanitize_openai_text(value: str, *, max_len: int | None = None) -> str:
    value = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_chars = []
    for char in value:
        if char in "\t\n":
            cleaned_chars.append(char)
            continue
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        cleaned_chars.append(char)
    value = "".join(cleaned_chars)
    if max_len is not None:
        value = value[:max_len]
    # Force an ASCII-safe payload so the downstream HTTP serializer never has to
    # deal with arbitrary Unicode emitted by very early checkpoints.
    return value.encode("ascii", errors="backslashreplace").decode("ascii")


def _sanitize_openai_messages(messages: list[dict]) -> list[dict]:
    sanitized_messages = []
    for message in messages:
        sanitized_message = dict(message)
        content = sanitized_message.get("content")
        if isinstance(content, str):
            sanitized_message["content"] = _sanitize_openai_text(content)
        sanitized_messages.append(sanitized_message)
    return sanitized_messages


def _validate_openai_messages(messages: list[dict]) -> None:
    json.dumps({"messages": messages}, ensure_ascii=False, allow_nan=False)



class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        self.model = model
        assert eval_type in ["base_0_100", "0_100", "0_10", "binary", "binary_text"], "eval_type must be either 0_100 or binary"
        self.eval_type = eval_type

        if self.eval_type == "0_100" or self.eval_type == "base_0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score
        else:
            raise ValueError(f"Invalid eval_type: {self.eval_type}")

        self.prompt_template = prompt_template
        
    async def judge(self, **kwargs):
        # Sanitize string values to prevent OpenAI API 400 JSON parse errors.
        # Very early checkpoints produce severely garbled text including:
        #   - C0 control chars (< 0x20) except tab/newline/CR
        #   - C1 control chars (0x80-0x9F) which are > " " but still invalid in JSON
        #   - Invalid Unicode surrogates that break UTF-8 serialization
        #   - Extremely long strings that exceed token limits
        clean_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = _sanitize_openai_text(
                    v,
                    max_len=MAX_ANSWER_LEN if k == "answer" else None,
                )
            clean_kwargs[k] = v
        prompt_content = _sanitize_openai_text(self.prompt_template.format(**clean_kwargs))
        messages = _sanitize_openai_messages([dict(role='user', content=prompt_content)])
        _validate_openai_messages(messages)
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(messages)
            score = self.aggregate_score(response_text) # aggregate_score is _aggregate_binary_text_score
        else:
            logprobs = await self.logprob_probs(messages)
            score = self.aggregate_score(logprobs) # aggregate_score is one of the other three
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await _chat_completion_with_retries(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))
        
        return result
    
    async def query_full_text(self, messages) -> str:
        """Requests a full text completion. Used for binary_text eval_type."""
        completion = await _chat_completion_with_retries(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=0
        )
        try:
            return completion.choices[0].message.content
        except (IndexError, AttributeError):
            return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
    def _aggregate_0_10_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explcitly.
        #   If "REFUSAL" has the highest probability, we return None.
        #   Otherwise, we aggregate scores from 0-9.
        if "REFUSAL" in score and score["REFUSAL"] > max(score.get(str(i), 0) for i in range(10)):
            return None

        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 9:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25: # Similar threshold as 0_100
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        #   NOTE: we check for refusals explicitly.
        #   If "REFUSAL" has a higher probability than both "YES" and "NO", we return None.
        #   Otherwise, calculates a score P(YES) / (P(YES) + P(NO)).
        #   Returns None if the combined probability of YES and NO is below a threshold (0.25).
        
        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        # If REFUSAL has a higher probability than both YES and NO, consider it a refusal.
        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None
        
        denominator = yes_prob + no_prob

        # If the combined probability of YES and NO is too low (e.g., model outputted something else,
        # or was not confident in YES/NO), return None.
        if denominator < 0.25:  # Using 0.25 to be consistent with other aggregation methods
            return None
            
        return yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> bool:
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None # Invalid response

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
