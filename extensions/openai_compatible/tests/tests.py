from openai import Client

client = Client(base_url="https://znhtl4bqsfyrv1e6.ai-plugin.io", api_key="local")

client.files.create()
completions = client.chat.completions.create(
    model="dify-chatflow-hello-world",
    messages=[{"role": "user", "content": "hello world"}],
    extra_body={"user_id": "sk-1"},
)

print(completions.choices[0].message.content)
