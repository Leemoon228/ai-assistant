import os
from pathlib import Path

from exceptions import GenerationError, UnknownIntentError
from services.classification import ClassificationService
from services import search_assistant


#OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_URL = "http://130.49.188.18:11434" # remote Ollama endpoint when Leemoon serves it
OLLAMA_MODEL = "gemma4:e4b"


class GenerateAnswerService:
    def __init__(self, classification_service: ClassificationService) -> None:
        self._search_assistant = search_assistant
        self._classification_service = classification_service

    def close(self) -> None:
        pass

    def warmup(self) -> None:
        self._search_assistant.warmup_model(
            model=OLLAMA_MODEL,
            ollama_url=OLLAMA_URL,
            timeout=float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")),
        )

    def generate(
        self,
        text: str,
        intent: str | None = None,
        order_context: str | None = None,
    ) -> dict[str, str | float | None]:
        prediction: dict[str, str | float] | None = None
        if not intent:
            prediction = self._classification_service.classify(text)
            intent = str(prediction["intent"])

        try:
            answer = self._search_assistant.answer_query(
                message=text,
                intent=intent,
                order_context=order_context,
                knowledge_path=(
                    Path(__file__).resolve().parents[1]
                    / "artifacts"
                    / "knowledge_db"
                    / "knowledge_db_v1.csv"
                ),
                model=OLLAMA_MODEL,
                ollama_url=OLLAMA_URL,
                timeout=float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")),
            )
        except ValueError as error:
            raise UnknownIntentError(str(error)) from error
        except RuntimeError as error:
            raise GenerationError(str(error)) from error

        return {
            "type": "instruction",
            "content": answer.answer,
            "follow_up_question": None,
            "intent": answer.intent,
            "category": answer.category,
            "model": answer.model,
            "classifier_model": prediction["model"] if prediction else None,
            "confidence": prediction["confidence"] if prediction else None,
            "classifier_duration_ms": prediction["duration_ms"] if prediction else None,
            "knowledge_lookup_ms": answer.knowledge_lookup_ms,
            "llm_duration_ms": answer.latency_ms,
        }
