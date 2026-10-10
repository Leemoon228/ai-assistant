from fastapi import APIRouter, Depends
from time import perf_counter

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


#  curl.exe -X POST http://localhost:8000/assistant/process -H "Content-Type: application/json" -d '{\"text\": \"Где найти кота\"}'
@router.post("/process", response_model=ProcessResponse)
def process(
    body: ProcessRequest,
    service: PreprocessingService = Depends(get_preprocessing_service),
):
    started = perf_counter()
    processed, truncated = service.process(body.text)
    return ProcessResponse(
        processed=processed,
        truncated=truncated,
        duration_ms=round((perf_counter() - started) * 1000, 3),
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    body: ClassifyRequest,
    service: ClassificationService = Depends(get_classification_service),
):
    return ClassifyResponse(**service.classify(body.text))


@router.post("/generate-answer", response_model=GenerateAnswerResponse)
def answer(
    body: GenerateAnswerRequest,
    service: GenerateAnswerService = Depends(get_generate_answer_service),
):
    started = perf_counter()
    result = service.generate(
        body.text,
        order_context=body.order_context,
    )
    result["duration_ms"] = round((perf_counter() - started) * 1000, 3)
    return GenerateAnswerResponse(
        **result
    )