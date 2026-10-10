# AI Assistant ML API

Инструкция для запуска Python/FastAPI сервиса AI Assistant.

Сервис выполняет предобработку текста, классификацию запроса и генерацию ответа через локальную Ollama.

## Что нужно
- Ollama и модель `gemma4:e4b` (см. установку ниже)
- Артефакты в этой папке:
  - `artifacts\classifier\models\classifier_v1.pt`
  - `artifacts\knowledge_db\knowledge_db_v1.csv`

## Установка Ollama и модели

Предполагается что Ollama установлен и доступен через bash

После установки откройте новое окно PowerShell и скачайте модель Gemma 4 E4B
из каталога Ollama:

```
ollama pull gemma4:e4b
ollama list
```

В выводе `ollama list` должна присутствовать `gemma4:e4b`.

## Запуск LLM

### 1. Проверить Ollama

Откройте отдельное окно терминала:

```
ollama list
```

В списке должна присутствовать `gemma4:e4b`. Обычно Ollama запускается
вместе с Windows. Если она не запущена, стартуйте её в отдельном окне:

```
ollama serve
```

Ollama должна быть доступна по адресу `http://127.0.0.1:11434`. Адрес и имя
модели задаются в `services\generating.py`.

### 2. Установить Python-зависимости

Из корня репозитория создайте виртуальное окружение:

```
python -m venv .venv
```

Активируйте его в используемом терминале:

```powershell
.\.venv\Scripts\Activate.ps1
```

```cmd
.\.venv\Scripts\activate.bat
```

```bash
source .venv/bin/activate
```

После активации установите зависимости один раз:

```
python -m pip install -r requirements.txt
```

### 3. Запустить ML API

В другом окне терминала из корня репозитория:

```
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Если открыли новое окно терминала, сначала снова активируйте `.venv`. Оставьте
окно с сервером открытым. Остановка сервиса: `Ctrl+C`.

### 4. Проверить API

В новом окне терминала:

```
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```
status
------
UP
```

Проверить генерацию ответа напрямую:

```
curl --% -X POST "http://127.0.0.1:8000/assistant/generate-answer" -H "Content-Type: application/json" -d "{\"text\":\"When are the next sales?\"}"
```

При генерации API автоматически определяет `intent` классификатором. Другие endpoints:

- `POST /assistant/process` — предобработка; JSON: `{"text":"..."}`.
- `POST /assistant/classify` — классификация; JSON: `{"text":"..."}`.
- `POST /assistant/generate-answer` — генерация ответа; JSON содержит `text`,
  необязательный `order_context`.

Swagger UI доступен по адресу `http://127.0.0.1:8000/docs`.

## Запуск всего приложения

Для работы сайта нужны три процесса: Ollama, этот ML API и backend из
`Market_AI_assistent_backend`. После запуска backend откройте
`http://localhost:5005/`.

