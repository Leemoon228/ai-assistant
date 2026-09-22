# pip install language-tool-python langdetect
from langdetect import detect, LangDetectException
import language_tool_python

from exceptions import UnsupportedLanguageError


class PreprocessingService:
    MAX_LENGTH = 200

    # def process0(self, text: str) -> tuple[str, bool]:
    #     # обрезка до 200 символов
    #     truncated = len(text) > self.MAX_LENGTH
    #     if truncated:
    #         text = text[: self.MAX_LENGTH]
    #
    #     return text, truncated

    def __init__(self):
        # LanguageTool по умолчанию использует публичный API
        # (лимиты: 20 запросов/мин, 75 000 символов/мин).
        # Для продакшена лучше поднять локальный сервер:
        #   language_tool_python.LanguageTool('en-US', config={'cacheSize': 1000})
        self._tool = language_tool_python.LanguageTool("en-US")

    def process(self, text: str) -> tuple[str, bool]:
        # 1. Проверка языка — до обрезки, чтобы не резать вслепую
        if not self._is_english(text):
            raise UnsupportedLanguageError(self._safe_detect(text))

        # 2. Обрезка до 200 символов
        truncated = len(text) > self.MAX_LENGTH
        if truncated:
            text = text[: self.MAX_LENGTH]

        # 3. Исправление опечаток и грамматики
        text = self._tool.correct(text)

        return text, truncated

    @staticmethod
    def _safe_detect(text: str) -> str | None:
        try:
            return detect(text)
        except LangDetectException:
            return None

    def _is_english(self, text: str) -> bool:
        # Пустая строка / пробелы — считаем валидной (нечего проверять)
        # !!! Вохможны ложные срабатывания на коротких тестах (пара слов)
        if not text or not text.strip():
            return True

        lang = self._safe_detect(text)
        return lang == "en"