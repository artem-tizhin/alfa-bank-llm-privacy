from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, FastAPI

from .config import Settings, get_settings
from .llm import LLMClient
from .ner import NERService
from .pipeline import PromptPipeline
from .schemas import MaskRequest, MaskResponse, ProcessRequest, ProcessResponse

router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> PromptPipeline:
    settings: Settings = get_settings()
    ner_service = NERService(
        model_path=settings.ner_model_path,
        device=settings.ner_device,
        max_length=settings.ner_max_length,
    )
    llm_client = LLMClient(settings)
    return PromptPipeline(ner_service=ner_service, llm_client=llm_client)


@router.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "ner_mode": pipeline.ner_service.mode,
        "llm_mode": pipeline.llm_client.mode,
        "llm_provider": pipeline.llm_client.provider,
    }


@router.post("/mask", response_model=MaskResponse)
def mask_endpoint(payload: MaskRequest) -> MaskResponse:
    return get_pipeline().mask_only(payload.text)


@router.post("/process", response_model=ProcessResponse)
def process_endpoint(payload: ProcessRequest) -> ProcessResponse:
    return get_pipeline().process(
        prompt=payload.prompt,
        system_prompt=payload.system_prompt,
        temperature=payload.temperature,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(router)
    return app