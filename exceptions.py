class AppError(Exception):
    """Базовое доменное исключение приложения."""
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class UnsupportedLanguageError(AppError):
    """Запрос пришёл на языке, отличном от английского."""
    status_code = 422
    code = "unsupported_language"

    def __init__(self, detected_lang: str | None = None):
        self.detected_lang = detected_lang
        message = (
           "Unsupported language. Only English is supported."
        )
        super().__init__(message)

class UnknownIntentError(AppError):
    """The requested intent has no matching knowledge article."""
    status_code = 430
    code = "unknown_intent"

class GenerationError(AppError):
    """The local language model could not generate a response."""
    status_code = 515
    code = "generation_failed"