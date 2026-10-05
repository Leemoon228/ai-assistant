from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from services.preprocessing import PreprocessingService
from services.classification import ClassificationService
from services.generating import GenerateAnswerService

from exceptions import AppError
from routers import router
# from http.http_client import close_client


# Если бдем использовать написанный http-клиент, его нужно закрывать при остановке fastApi
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     await close_client()

# Запуск: uvicorn main:app --reload

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.preprocessing_service = PreprocessingService()
    app.state.classification_service = ClassificationService()
    app.state.generate_answer_service = GenerateAnswerService()
    try:
        yield
    finally:
        app.state.preprocessing_service.close()
        app.state.classification_service.close()
        app.state.generate_answer_service.close()


# TODO: Добавить возможность задать порт
app = FastAPI(lifespan=lifespan, title="AI Assistant ML Service", version="0.1.0") #, lifespan=lifespan)
app.include_router(router)

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )

@app.get("/health")
def health():
    return {"status": "UP"}