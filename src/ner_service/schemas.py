from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntitySpan(BaseModel):
    start: int
    end: int
    label: str
    text: str


class MaskMapping(BaseModel):
    placeholder: str
    original_text: str
    label: str
    start: int
    end: int


class MaskRequest(BaseModel):
    text: str = Field(..., description="Исходный текст")


class MaskResponse(BaseModel):
    original_text: str
    masked_text: str
    entities: List[EntitySpan]
    mapping: List[MaskMapping]


class ProcessRequest(BaseModel):
    prompt: str = Field(..., description="Исходный prompt пользователя")
    system_prompt: Optional[str] = Field(default=None, description="System prompt для LLM")
    temperature: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    original_prompt: str
    masked_prompt: str
    llm_masked_response: str
    final_response: str
    entities: List[EntitySpan]
    mapping: List[MaskMapping]
    llm_mode: str
