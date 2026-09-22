# Заготовка под http-клиент для вызова внешних апи

# import httpx
#
# _client: httpx.AsyncClient | None = None
#
#
# async def get_client() -> httpx.AsyncClient:
#     """Вернуть общий AsyncClient, создав его при первом вызове."""
#     global _client
#     if _client is None:
#         _client = httpx.AsyncClient(
#             timeout=httpx.Timeout(30.0, connect=5.0),
#             limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
#             headers={"User-Agent": "ai-assistant-ml/0.1.0"},
#         )
#     return _client
#
#
# async def close_client() -> None:
#     """Закрыть клиент при остановке приложения."""
#     global _client
#     if _client is not None:
#         await _client.aclose()
#         _client = None