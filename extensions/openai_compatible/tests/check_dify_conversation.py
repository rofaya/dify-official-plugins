import httpx

# base_url = "https://api.dify.ai/v1"
# api_key = "app-SP1SmK29AQqJ3GKw9GXqcCGO"

base_url = "http://192.168.1.180/v1"
api_key = "app-BjHU6K7ppeC9sVx7fCFIXLqS"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

client = httpx.Client(base_url=base_url, headers=headers)

# payload = {
#     "inputs": {},
#     "query": "hello",
#     "response_mode": "streaming",
#     "conversation_id": "",
#     "user": "0124c5f0-7079-4b39-8268-41b9699452df",
#     "files": [],
# }
#
# response = client.post("/chat-messages", json=payload)

params = {"user": "0124c5f0-7079-4b39-8268-41b9699452df"}
conversations = client.get("/conversations", params=params)
print(conversations.json())

# params = {
#     "conversation_id": "9d27044a-c25e-4db9-942e-c92f6308fd46",
#     "user": "",
#     "first_id": None,
#     "limit": 20,
# }
# messages = client.get("/messages", params=params)
# print(messages.json())
