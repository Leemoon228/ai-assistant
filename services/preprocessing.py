
# pip install fasttext-langdetect text-corrector

import logging
from time import perf_counter

from ftlangdetect import detect
from text_corrector import correct_text

from exceptions import UnsupportedLanguageError


logger = logging.getLogger(__name__)


class PreprocessingService:
    MAX_LENGTH = 200

    def __init__(self):
        pass

    def warmup(self) -> None:
        """Прогревает детектор языка и корректор перед первым запросом."""
        started = perf_counter()

        try:
            t0 = perf_counter()
            detect(
                text="This is a warmup sentence for language detection.",
                low_memory=True,
            )
            logger.info(
                "Language detection warmup: %.3fs",
                perf_counter() - t0,
            )

            t1 = perf_counter()
            correct_text(
                "This is a warmup sentence.",
                max_dist=2,
            )
            logger.info(
                "Text correction warmup: %.3fs",
                perf_counter() - t1,
            )

            logger.info(
                "Preprocessing warmup completed in %.3fs",
                perf_counter() - started,
            )
        except Exception:
            logger.exception("Preprocessing warmup failed")
            raise

    def close(self) -> None:
        # Закрывать нечего — ресурсов больше нет.
        # Метод оставлен для совместимости с lifespan.
        pass

    def process(self, text: str) -> tuple[str, bool]:
        # Пустую строку не нужно отправлять в детектор.
        if not text or not text.strip():
            return text, False

        # 1. Проверка языка — до обрезки.
        lang = self._safe_detect(text)
        if lang != "en":
            raise UnsupportedLanguageError(lang)

        # 2. Обрезка до 200 символов.
        truncated = len(text) > self.MAX_LENGTH
        if truncated:
            text = text[:self.MAX_LENGTH]

        # 3. Исправление опечаток.
        text, _ = correct_text(text, max_dist=2)

        return text, truncated

    @staticmethod
    def _safe_detect(text: str) -> str | None:
        try:
            result = detect(text=text, low_memory=True)
            return result["lang"]
        except Exception:
            logger.exception("Language detection failed")
            return None

    def _is_english(self, text: str) -> bool:
        # Пустая строка / пробелы — считаем валидными.
        if not text or not text.strip():
            return True

        return self._safe_detect(text) == "en"
