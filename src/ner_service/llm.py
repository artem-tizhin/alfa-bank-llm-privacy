from __future__ import annotations

from typing import Optional

from .config import Settings

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = settings.llm_mode
        self.provider = "stub"
        self._client = None

        if self.mode == "groq":
            self.provider = "groq"

            if not settings.groq_api_key:
                raise RuntimeError(
                    "LLM_MODE='groq', но не задан GROQ_API_KEY"
                )

            if Groq is None:
                raise RuntimeError(
                    "Не установлен пакет groq. Установи зависимости из requirements.txt"
                )

            self._client = Groq(
                api_key=settings.groq_api_key,
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.mode == "stub":
            return prompt

        if self.mode != "groq" or self._client is None:
            raise RuntimeError(
                f"Неподдерживаемый llm_mode='{self.mode}' или клиент не инициализирован"
            )

        response = self._client.chat.completions.create(
            model=self.settings.groq_model,
            temperature=self.settings.groq_temperature if temperature is None else temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    or (
                        "Ты помощник. "
                        "Сохраняй все плейсхолдеры вида [EMAIL_0], [PHONE_0], [PASSPORT_0] "
                        "без изменений. Не удаляй их, не переименовывай и не искажай."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content
        return content or ""