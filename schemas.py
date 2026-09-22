from pydantic import BaseModel, Field
from typing import Literal, Optional


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Пользовательский запрос")


class ProcessResponse(BaseModel):
    processed: str
    truncated: bool


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Обработанный запрос")


class ClassifyResponse(BaseModel):
    category: str
    confidence: float


class GenerateAnswerRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Запрос пользователя")
    category: Optional[str] = Field(None, description="Опционально - категория из шага 2")


class GenerateAnswerResponse(BaseModel):
    type: Literal["instruction", "clarification"]
    content: str
    follow_up_question: Optional[str] = None