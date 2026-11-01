import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INDEX_BASE = r"D:\marketing-agent\index"

print("正在加载 Embedding 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 缓存已加载的索引
_index_cache = {}


def load_index(brand_name):
    """加载指定品牌的FAISS索引（带缓存）"""
    if brand_name in _index_cache:
        return _index_cache[brand_name]

    index_dir = os.path.join(INDEX_BASE, brand_name)
    index_path = os.path.join(index_dir, "faiss_index.bin")
    texts_path = os.path.join(index_dir, "texts.txt")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"品牌 {brand_name} 的索引不存在，请先运行 build_index.py")

    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts_data = [line.strip().split("||", 1) for line in f if line.strip()]

    _index_cache[brand_name] = (index, texts_data)
    return index, texts_data


def retrieve(query, brand_name="olay", top_k=3):
    """检索指定品牌的知识库"""
    index, texts_data = load_index(brand_name)

    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i in indices[0]:
        if i != -1:
            results.append(texts_data[i])
    return results


def generate_marketing_copy(user_query, retrieved_docs):
    context = "\n".join([f"[{doc[0]}] {doc[1]}" for doc in retrieved_docs])

    prompt = f"""你是一名专业的营销文案专家。请根据以下品牌指南和产品信息，为用户生成一条营销文案。

【品牌与产品信息】
{context}

【用户需求】
{user_query}

【输出要求】
1. 严格遵守品牌语气和禁用词要求。
2. 贴合小红书/抖音等平台的网感。
3. 直接输出文案正文，不需要解释过程。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业且严格遵守品牌指南的营销文案生成助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
	 timeout=60
    )
    return response.choices[0].message.content