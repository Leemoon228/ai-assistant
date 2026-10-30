import logging
import uuid
from time import perf_counter

from fastapi import APIRouter, Depends, Header, Response

from schemas import (
    ProcessRequest, ProcessResponse,
    ClassifyRequest, ClassifyResponse,
    GenerateAnswerRequest, GenerateAnswerResponse,
)

from dependencies import (
    get_preprocessing_service,
    get_classification_service,
    get_generate_answer_service,
)
from services.preprocessing import PreprocessingService
from services.classification import ClassificationService
from services.generating import GenerateAnswerService

router = APIRouter(prefix="/assistant", tags=["assistant"])
logger = logging.getLogger("uvicorn.error")


def resolve_trace_id(x_trace_id: str | None, response: Response) -> str:
    """Переиспользует X-Trace-Id от вызывающей стороны (backend) или создаёт
    новый — так лог одного запроса можно найти и в backend, и здесь."""
    trace_id = x_trace_id or str(uuid.uuid4())
    response.headers["X-Trace-Id"] = trace_id
    return trace_id


#  curl.exe -X POST http://localhost:8000/assistant/process -H "Content-Type: application/json" -d '{\"text\": \"Где найти кота\"}'
@router.post("/process", response_model=ProcessResponse)
def process(
    body: ProcessRequest,
    response: Response,
    x_trace_id: str | None = Header(default=None),
    service: PreprocessingService = Depends(get_preprocessing_service),
):
    trace_id = resolve_trace_id(x_trace_id, response)
    started = perf_counter()
    processed, truncated = service.process(body.text)
    duration_ms = round((perf_counter() - started) * 1000, 3)
    logger.info("[%s] process done (%sms)", trace_id, duration_ms)
    return ProcessResponse(
        processed=processed,
        truncated=truncated,
        duration_ms=duration_ms,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    body: ClassifyRequest,
    response: Response,
    x_trace_id: str | None = Header(default=None),
    service: ClassificationService = Depends(get_classification_service),
):
    trace_id = resolve_trace_id(x_trace_id, response)
    result = service.classify(body.text)
    logger.info(
        "[%s] classify -> intent=%s confidence=%.2f (%sms)",
        trace_id, result.get("intent"), result.get("confidence", 0.0), result.get("duration_ms"),
    )
    return ClassifyResponse(**result)


@router.post("/generate-answer", response_model=GenerateAnswerResponse)
def answer(
    body: GenerateAnswerRequest,
    response: Response,
    x_trace_id: str | None = Header(default=None),
    service: GenerateAnswerService = Depends(get_generate_answer_service),
):
    trace_id = resolve_trace_id(x_trace_id, response)
    started = perf_counter()
    result = service.generate(
        body.text,
        order_context=body.order_context,
    )
    result["duration_ms"] = round((perf_counter() - started) * 1000, 3)
    logger.info(
        "[%s] generate-answer -> model=%s (%sms)",
        trace_id, result.get("model"), result["duration_ms"],
    )
    return GenerateAnswerResponse(
        **result
    )
