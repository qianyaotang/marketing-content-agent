import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = r"D:\marketing-agent\data"
INDEX_BASE = r"D:\marketing-agent\index"

print("正在加载 Embedding 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def build_index_for_brand(brand_name, brand_dir):
    """为单个品牌构建FAISS索引"""
    docs = []
    for filename in os.listdir(brand_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(brand_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = [line.strip() for line in text.split("\n") if line.strip()]
                for chunk in chunks:
                    docs.append({"text": chunk, "source": f"{brand_name}/{filename}"})

    if not docs:
        print(f"  ⚠️ {brand_name}: 没有找到数据")
        return

    texts = [doc["text"] for doc in docs]
    print(f"  {brand_name}: 共加载 {len(texts)} 条文本块")

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_dir = os.path.join(INDEX_BASE, brand_name)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss_index.bin"))

    with open(os.path.join(index_dir, "texts.txt"), "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(f"{doc['source']}||{doc['text']}\n")

    print(f"  ✅ {brand_name}: 索引构建成功（{len(texts)}条，维度{dimension}）")


if __name__ == "__main__":
    # 遍历 data 下的品牌文件夹（排除根目录的全局文件）
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            print(f"\n📦 正在处理品牌：{item}")
            build_index_for_brand(item, item_path)

    print("\n🎉 所有品牌索引构建完成！")