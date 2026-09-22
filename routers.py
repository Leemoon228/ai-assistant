from fastapi import APIRouter

from schemas import (
    ProcessRequest, ProcessResponse,
    ClassifyRequest, ClassifyResponse,
    GenerateAnswerRequest, GenerateAnswerResponse,
)

from services.preprocessing import PreprocessingService
from services.classification import ClassificationService
from services.generating import GenerateAnswerService

router = APIRouter(prefix="/assistant", tags=["assistant"])

#  curl.exe -X POST http://localhost:8000/assistant/process -H "Content-Type: application/json" -d '{\"text\": \"Где найти кота\"}'
@router.post("/process", response_model=ProcessResponse)
def process(body: ProcessRequest):
    service = PreprocessingService()
    processed, truncated = service.process(body.text)
    return ProcessResponse(
        processed=processed,
        truncated=truncated,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest):
    service = ClassificationService()
    return ClassifyResponse(**service.classify(body.text))


@router.post("/generate-answer", response_model=GenerateAnswerResponse)
def answer(body: GenerateAnswerRequest):
    service = GenerateAnswerService()
    return GenerateAnswerResponse(**service.generate(body.text, body.category))