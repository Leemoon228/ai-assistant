# AI Assistant ML API

Python/FastAPI сервис для предобработки текста, классификации запросов и генерации ответов с помощью Ollama.

## Требования

- Python 3.12 (версия указана в `.python-version`) для локального запуска
- Ollama и модель `gemma4:e4b`
- Артефакты проекта:
  - `artifacts\classifier\models\classifier_v1.pt`
  - `artifacts\knowledge_db\knowledge_db_v1.csv`

## Локальный запуск без Docker

Для локальной разработки установите Python 3.12 (версия указана в
`.python-version`) и создайте виртуальное окружение:

```powershell
python -m venv .venv
```

Активируйте окружение командой для используемой оболочки:

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows cmd:
```cmd
.\.venv\Scripts\activate.bat
```

Bash:
```bash
source .venv/Scripts/activate
```

Установить зависимости:
```powershell
python -m pip install -r requirements.txt
```

Запустите Ollama локально и убедитесь, что модель `gemma4:e4b` доступна:

```powershell
ollama serve
ollama pull gemma4:e4b
ollama list
```

Затем запустите ML API из корня репозитория:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Если ошибка preprocessing, то нужно использовать кэш библиотеки fasttext из папки ./fasttext-cache перед запуском:

Powershell:
```powershell
$env:FTLANG_CACHE = ".\fasttext-cache"
```

Локальный запуск проверяет Ollama на `http://127.0.0.1:11434`, затем пробует
резервный адрес (его хостит @Leemoon2 удалённо). Переопределить адреса можно переменными `OLLAMA_BASE_URL` и
`OLLAMA_FALLBACK_URL`.

## Запуск через Docker Compose

Обычная команда запускает только ML API и **не запускает Ollama-контейнер**:

```powershell
docker compose up -d --build
```
Лога:
```
docker compose logs -f ai-assistant
```

В этом режиме API подключается к Ollama на хост-компьютере по адресу
`http://host.docker.internal:11434`. Установите и запустите Ollama на хосте,
если используете этот вариант.

### Ollama в Docker (не советую)
Сначала запустите только контейнер Ollama и загрузите модель. Ollama будет доступна на хосте по адресу `http://127.0.0.1:11434`:

Контейнеру потребуется около 8GB RAM. При CPU сборке на 8 ядер, LLM отвечает 10+ сек.

По-хорошему подвязать использование GPU из контейнера. Тут есть инфа как это сделать [docker ollama repo](https://hub.docker.com/r/ollama/ollama). Нужно либо установить nvidia container toolkit, либо для amd (linux only) запускать образ докера ollama с тегом rocm

CPU only установка (примерно n секунд на генерацию ответа):
```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
После установки ollama, нужно скачать в неё LLM модель.
```
docker exec -it ollama ollama pull gemma4:e4b
docker exec -it ollama ollama list
```

API-контейнер обращается к Ollama через локалку 127.0.0.1 на хосте
`http://127.0.0.0:11434`

Для просмотра логов:
```powershell
docker logs -f ollama
```

Образ Ollama: [`ollama/ollama` на Docker Hub](https://hub.docker.com/r/ollama/ollama).
Модель `gemma4:e4b` хранится в Docker volume `ollama_data` и не загружается
заново при обычном перезапуске контейнера.



## Проверка API

API доступен по адресу `http://127.0.0.1:8000`; Swagger UI —
`http://127.0.0.1:8000/docs`.

Проверка состояния:

```powershell
curl http://127.0.0.1:8000/health
```

Предобработка текста:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/assistant/process" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"When are the next sales?\"}"
```

Классификация запроса:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/assistant/classify" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"When are the next sales?\"}"
```

Генерация ответа:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/assistant/generate-answer" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"When are the next sales?\"}"
```

При генерации API автоматически определяет `intent` с помощью классификатора.
Для `POST /assistant/generate-answer` можно также передать необязательное поле
`order_context` с проверенными сведениями о заказе.