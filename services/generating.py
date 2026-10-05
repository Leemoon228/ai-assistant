class GenerateAnswerService:
    def close(self) -> None:
        # TODO: реализовать закрытие необходимых ресурсов
        pass

    def generate(self, text: str, category: str | None = None) -> dict:
        # TODO: вызов основной ML-модели (выдача инструкции или уточнения)

        return {
            "type": "clarification",
            "content": "Заглушка ответа.",
            "follow_up_question": "Уточните, пожалуйста, ваш запрос.",
        }
