import json
import logging

import groq
from pydantic import ValidationError

from app.core.config import get_settings
from app.prompts.assistant_prompts import assistant_system
from app.prompts.review_prompts import REVIEW_SYSTEM
from app.prompts.test_prompts import TEST_SYSTEM
from app.schemas.schemas import AIReview, ChatRequest, CodeRequest, FixPlan, FixVerification
from app.prompts.review_prompts import SAFETY


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class GroqService:
    def _client(self):
        config = get_settings()
        if not config.groq_api_key:
            raise ServiceError(
                "Set GROQ_API_KEY in backend/.env and restart the backend.", 503
            )
        return groq.AsyncGroq(
            api_key=config.groq_api_key, timeout=config.provider_timeout, max_retries=0
        )

    async def _complete(self, messages: list[dict], structured: bool = False) -> str:
        options = {"response_format": {"type": "json_object"}} if structured else {}
        try:
            async with self._client() as client:
                result = await client.chat.completions.create(
                    model=get_settings().groq_model,
                    messages=messages,
                    temperature=0.15,
                    max_completion_tokens=10000,
                    **options,
                )
            content = result.choices[0].message.content
            if not content or result.choices[0].finish_reason == "length":
                raise ServiceError(
                    "Groq returned empty or truncated output. Try a smaller input."
                )
            return content
        except groq.AuthenticationError:
            raise ServiceError(
                "Groq rejected the backend API key. Update backend/.env.", 503
            ) from None
        except groq.RateLimitError:
            raise ServiceError(
                "Groq rate limit reached. Please retry shortly.", 429
            ) from None
        except groq.APITimeoutError:
            raise ServiceError(
                "Groq request timed out. Please retry with less code.", 504
            ) from None
        except groq.APIConnectionError:
            raise ServiceError(
                "Cannot connect to Groq. Check backend network access.", 503
            ) from None
        except groq.APIStatusError as error:
            # Log only status and a known error category, never provider bodies:
            # failed generations can contain submitted source or other secrets.
            logging.getLogger(__name__).warning(
                "Groq request rejected: HTTP %s",
                error.status_code,
            )
            if error.status_code == 413:
                raise ServiceError(
                    "Groq rejected this review as too large for the account limits. "
                    "Try less code or a smaller pull request.",
                    413,
                ) from None
            raise ServiceError(
                "Groq could not complete the request. Check GROQ_MODEL and account access."
            ) from None

    async def review(self, payload: dict) -> AIReview:
        messages = [
            {
                "role": "system",
                "content": REVIEW_SYSTEM
                + "\nJSON schema:\n"
                + json.dumps(AIReview.model_json_schema()),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        for attempt in range(2):
            content = await self._complete(messages, structured=True)
            try:
                return AIReview.model_validate_json(content)
            except ValidationError:
                if attempt:
                    raise ServiceError(
                        "Groq returned invalid review data after one repair attempt. Please retry."
                    ) from None
                messages.append(
                    {
                        "role": "user",
                        "content": "The response did not match the schema. Return a fresh valid JSON object matching every field and enum. Do not add commentary.",
                    }
                )
        raise ServiceError("No validated review returned.")

    async def chat(self, request: ChatRequest) -> str:
        return await self._complete(
            [
                {"role": "system", "content": assistant_system(request.action)},
                *[m.model_dump() for m in request.messages],
            ]
        )

    async def tests(self, request: CodeRequest) -> str:
        return await self._complete(
            [
                {"role": "system", "content": TEST_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "language": request.language,
                            "code": request.code,
                            "instructions": request.instructions,
                        }
                    ),
                },
            ]
        )

    async def connection_status(self) -> tuple[bool, str]:
        if not get_settings().groq_api_key:
            return False, "Not configured"
        try:
            async with self._client() as client:
                await client.models.retrieve(get_settings().groq_model)
            return True, "Connected (model accessible)"
        except groq.AuthenticationError:
            return False, "Invalid API key"
        except groq.APIError:
            return False, "Connection or model access failed"

    async def _fix_json(self, instruction: str, payload: dict, schema):
        messages = [
            {"role": "system", "content": SAFETY + "\n" + instruction
             + "\nReturn JSON matching this schema: " + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": json.dumps(payload)},
        ]
        for attempt in range(2):
            content = await self._complete(messages, structured=True)
            try:
                return schema.model_validate_json(content)
            except ValidationError:
                if attempt:
                    raise ServiceError("Invalid fix response after one schema repair attempt.") from None
                messages.append({"role": "user", "content": "Return valid JSON matching every schema field and enum. No commentary."})

    async def generate_fixes(self, payload: dict) -> FixPlan:
        return await self._fix_json(
            "Correct ONLY the identified, supported issues using the exact original files and supplied context. "
            "Preserve business logic, public interfaces, comments and unrelated formatting. "
            "Return complete corrected source, without Markdown fences, for changed existing files only. "
            "Use the supplied zero-based issue IDs. Skip uncertain, unsupported or unlocated findings with a reason. "
            "Do not change dependency manifests, workflows, secrets, tests, or invent paths. "
            "Do not remove functionality to silence a finding. Use validation feedback to repair an invalid prior attempt.",
            payload, FixPlan,
        )

    async def verify_fixes(self, payload: dict) -> FixVerification:
        return await self._fix_json(
            "Independently audit original and corrected source against each proposed issue ID and its evidence. "
            "Do not trust the fix generator's claims. Mark unsupported or uncertain issues explicitly. "
            "Report unintended changes or lost business logic; behavior_preserved means unrelated behavior and interfaces remain intact. "
            "A fixed verdict requires concrete source evidence. This is an AI assessment, never a claim of test execution.",
            payload, FixVerification,
        )
