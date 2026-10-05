from fastapi import Request

from services.preprocessing import PreprocessingService
from services.classification import ClassificationService
from services.generating import GenerateAnswerService

def get_preprocessing_service(request: Request) -> PreprocessingService:
    return request.app.state.preprocessing_service


def get_classification_service(request: Request) -> ClassificationService:
    return request.app.state.classification_service


def get_generate_answer_service(request: Request) -> GenerateAnswerService:
    return request.app.state.generate_answer_service