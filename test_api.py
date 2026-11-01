import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "用一句话介绍你自己"}
        ]
    )
    print("API调用成功！")
    print("模型回复：", response.choices[0].message.content)
except Exception as e:
    print("API调用失败，错误信息：")
    print(e)