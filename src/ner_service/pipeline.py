from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm import LLMClient
from .masking import mask_text, spans_to_entities, unmask_text
from .ner import NERService
from .schemas import MaskResponse, ProcessResponse


@dataclass
class PromptPipeline:
    ner_service: NERService
    llm_client: LLMClient

    def mask_only(self, text: str) -> MaskResponse:
        spans = self.ner_service.predict(text)
        entities = spans_to_entities(text, spans)
        masked_text, mapping = mask_text(text, spans)
        return MaskResponse(
            original_text=text,
            masked_text=masked_text,
            entities=entities,
            mapping=mapping,
        )

    def process(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> ProcessResponse:
        masked = self.mask_only(prompt)
        llm_response_masked = self.llm_client.generate(
            prompt=masked.masked_text,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        final_response = unmask_text(llm_response_masked, masked.mapping)
        return ProcessResponse(
            original_prompt=prompt,
            masked_prompt=masked.masked_text,
            llm_masked_response=llm_response_masked,
            final_response=final_response,
            entities=masked.entities,
            mapping=masked.mapping,
            llm_mode=self.llm_client.mode,
        )
