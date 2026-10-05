from fastapi import APIRouter, Depends

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
    processed, truncated = service.process(body.text)
    return ProcessResponse(
        processed=processed,
        truncated=truncated,
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
    return GenerateAnswerResponse(**service.generate(body.text, body.category))