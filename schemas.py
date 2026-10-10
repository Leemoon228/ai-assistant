from pydantic import BaseModel, Field
from typing import Literal, Optional


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Пользовательский запрос")


class ProcessResponse(BaseModel):
    processed: str
    truncated: bool
    duration_ms: float


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Обработанный запрос")


class ClassifyResponse(BaseModel):
    category: str
    intent: str
    confidence: float
    model: str
    duration_ms: float


class GenerateAnswerRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Запрос пользователя")
    order_context: Optional[str] = Field(None, description="Проверенные данные заказа из backend")


class GenerateAnswerResponse(BaseModel):
    type: Literal["instruction", "clarification"]
    content: str
    follow_up_question: Optional[str] = None
    intent: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    classifier_model: Optional[str] = None
    confidence: Optional[float] = None
    classifier_duration_ms: Optional[float] = None
    knowledge_lookup_ms: Optional[float] = None
    llm_duration_ms: Optional[float] = None
    duration_ms: float