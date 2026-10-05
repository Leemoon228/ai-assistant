# pip install fasttext-langdetect text-corrector
from ftlangdetect import detect
from text_corrector import correct_text

from exceptions import UnsupportedLanguageError


class PreprocessingService:
    MAX_LENGTH = 200

    def __init__(self):
        # LanguageTool удалён из флоу.
        # Осталась только пословная коррекция опечаток через text_corrector.
        pass

    def close(self) -> None:
        # Закрывать нечего — ресурсов больше нет.
        # Метод оставлен для совместимости с lifespan.
        pass

    def process(self, text: str) -> tuple[str, bool]:
        # 1. Проверка языка — до обрезки, чтобы не резать вслепую
        if not self._is_english(text):
            raise UnsupportedLanguageError(self._safe_detect(text))

        # 2. Обрезка до 200 символов
        truncated = len(text) > self.MAX_LENGTH
        if truncated:
            text = text[: self.MAX_LENGTH]

        # 3. Исправление опечаток (пословно, на основе словаря)
        text, _ = correct_text(text, max_dist=2)

        return text, truncated

    @staticmethod
    def _safe_detect(text: str) -> str | None:
        try:
            result = detect(text=text, low_memory=False)
            return result["lang"]
        except Exception:
            return None

    def _is_english(self, text: str) -> bool:
        # Пустая строка / пробелы — считаем валидной (нечего проверять)
        # Возможны ложные срабатывания на коротких текстах (пара слов)
        if not text or not text.strip():
            return True

        lang = self._safe_detect(text)
        return lang == "en"