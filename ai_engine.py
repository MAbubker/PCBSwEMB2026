import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 🔥 تحميل موديل embedding
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

df_global = None
index = None
embeddings = None

def load_data(df):
    global df_global, index, embeddings

    # 🔥 توحيد أسماء الأعمدة
    if "description" not in df.columns:
        if "activity_description" in df.columns:
            df = df.rename(columns={"activity_description": "description"})
        else:
            raise ValueError("❌ يجب وجود عمود description أو activity_description")

    if "code" not in df.columns:
        raise ValueError("❌ يجب وجود عمود code")

    df["description"] = df["description"].astype(str)

    df_global = df.reset_index(drop=True)

    texts = df_global["description"].tolist()

    # 🔥 إنشاء embeddings مرة واحدة
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)


def search(text, top_n=5):
    global df_global, index

    emb = embed_model.encode([text]).astype("float32")
    D, I = index.search(emb, top_n)

    results = df_global.iloc[I[0]].copy()
    results["score"] = 1 / (1 + D[0])

    return results