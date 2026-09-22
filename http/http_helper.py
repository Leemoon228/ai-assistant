# Заготовка под методы http-клиента для вызова внешних апи

# from http.http_client import get_client
#
#
# async def get_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
#     client = await get_client()
#     response = await client.get(url, params=params, headers=headers)
#     response.raise_for_status()
#     return response.json()