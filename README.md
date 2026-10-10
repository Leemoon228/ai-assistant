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

Для проекта используется Python 3.12 (версия указана в `.python-version`).
PyTorch устанавливается в CPU-сборке.

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

Windows powershell
```
curl --% -X POST "http://127.0.0.1:8000/assistant/generate-answer" -H "Content-Type: application/json" -d "{\"text\":\"When are the next sales?\"}"
```

При генерации API автоматически определяет `intent` классификатором. Другие endpoints:

- `POST /assistant/process` — предобработка; JSON: `{"text":"..."}`.
- `POST /assistant/classify` — классификация; JSON: `{"text":"..."}`.
- `POST /assistant/generate-answer` — генерация ответа; JSON содержит `text`,
  необязательный `order_context`.

Swagger UI доступен по адресу `http://127.0.0.1:8000/docs`.

## Запуск AI-Assistant через Docker Compose

Сервис можно запустить двумя способами:
- с Ollama, установленной локально;
- с Ollama в отдельном Docker-контейнере.

## Вариант 1. Ollama установлена локально на Windows

В этом варианте Ollama запускается локально, а AI-ассистент работает в Docker-контейнере. Контейнер использует уже установленную модель, поэтому Ollama не нужно запускать через Docker Compose.

### 1. Установка Ollama и загрузка модели (см. инструкцию ранее)
### 2. Настройка Docker Compose

В `docker-compose.yml` используйте конфигурацию с OLLAMA_BASE_URL: http://host.docker.internal:11434

### 4. Запуск AI-ассистента

Из каталога проекта выполните:

```powershell
docker compose up -d --build
```

Проверьте логи:

```powershell
docker compose logs -f ai-assistant
```

После запуска API будет доступен по адресу:

- Swagger UI: http://localhost:8085/docs

### 5. Если Ollama недоступна из контейнера

Убедитесь, что:

- Ollama запущена и отвечает на `http://localhost:11434/api/tags` в Windows.
- В контейнер передаётся `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
- Код приложения действительно использует переменную `OLLAMA_BASE_URL`, а не обращается к `127.0.0.1:11434` напрямую.

### Вариант 2. Ollama в отдельном Docker-контейнере

В этом режиме приложение и Ollama запускаются совместно через Docker Compose. Модель загружается автоматически при первом запуске и сохраняется в Docker volume.

### 1. Замените конфигурацию из первого варианта закоменченным блоком в `docker-compose.yml`.
### 2. Запустите сервисы:

```bash
docker compose up -d --build
```

### 3. ### Проверьте состояние контейнеров и логи:

```bash
docker compose ps
docker compose logs -f ai-assistant
docker compose logs -f ollama
```

При первом запуске Ollama скачает модель `gemma4:e4b`. Время загрузки зависит от скорости интернета и размера модели.

### 4. Проверка работы

После запуска API будет доступен по адресу:

- Swagger UI: http://localhost:8085/docs
- OpenAPI: http://localhost:8085/openapi.json

Для остановки сервисов:

```bash
docker compose down
```

При остановке Docker volume с моделью сохраняется. Чтобы удалить его вместе с загруженными моделями, выполните:

```bash
docker compose down -v
```

## Запуск всего приложения

Для работы сайта нужны три процесса: Ollama, этот ML API и backend из
`Market_AI_assistent_backend`. После запуска backend откройте
`http://localhost:5005/`.
