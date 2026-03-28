import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# تحميل النموذج مرة واحدة فقط
# ==========================================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# تحميل بيانات التصنيف
# ==========================================
CLASS_FILE = "data\ISIC04-2014.xls"
FEEDBACK_FILE = "feedback.xlsx"

df_class = pd.read_excel(CLASS_FILE)

# تنظيف
df_class["description"] = df_class["description"].astype(str)

# embeddings جاهزة
class_embeddings = model.encode(df_class["description"].tolist(), show_progress_bar=False)

# ==========================================
# تنظيف النص
# ==========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

# ==========================================
# استخراج كلمات مهمة من اسم المنشأة
# ==========================================
def extract_keywords(text):
    if text is None:
        return ""
    words = str(text).split()
    return " ".join(words[:3])  # أول 3 كلمات

# ==========================================
# similarity باستخدام embedding جاهز
# ==========================================
def predict_similarity_from_embedding(emb, top_n=5):
    sims = cosine_similarity([emb], class_embeddings)[0]
    idxs = sims.argsort()[-top_n:][::-1]

    results = df_class.iloc[idxs].copy()
    results["score"] = sims[idxs]

    return results

# ==========================================
# classifier (بسيط من similarity)
# ==========================================
def predict_classifier(emb):
    sims = cosine_similarity([emb], class_embeddings)[0]
    idx = sims.argmax()
    return df_class.iloc[idx]["code"], sims[idx]

# ==========================================
# تحميل feedback
# ==========================================
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_excel(FEEDBACK_FILE)
        df["activity"] = df["activity"].astype(str)
        return df
    return pd.DataFrame(columns=["activity", "code"])

# ==========================================
# البحث في feedback
# ==========================================
def get_feedback_match(text):
    df = load_feedback()
    if df.empty:
        return None

    text = str(text)

    matches = df[df["activity"].str.contains(text[:15], na=False)]
    if not matches.empty:
        return matches.iloc[-1]["code"]

    return None

# ==========================================
# حفظ التعلم
# ==========================================
def save_feedback(activity, code, original_text=""):
    os.makedirs(".", exist_ok=True)

    new_data = pd.DataFrame([{
        "activity": str(activity),
        "code": str(code),
        "original": str(original_text)
    }])

    if os.path.exists(FEEDBACK_FILE):
        old = pd.read_excel(FEEDBACK_FILE)
        df = pd.concat([old, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_excel(FEEDBACK_FILE, index=False)

# ==========================================
# النظام الهجين النهائي
# ==========================================
def hybrid_predict(text, emb, top_n=5):

    # 1. تعلم من المستخدم
    fb_code = get_feedback_match(text)
    if fb_code:
        return fb_code, 1.0, "تعلم من المستخدم"

    # 2. similarity
    top_matches = predict_similarity_from_embedding(emb, top_n)
    best_sim = top_matches.iloc[0]

    # 3. classifier
    clf_code, clf_score = predict_classifier(emb)

    # 4. دمج القرار
    if best_sim["score"] > 0.75:
        return best_sim["code"], best_sim["score"], "Similarity قوي"

    elif clf_score > 0.6:
        return clf_code, clf_score, "Classifier"

    else:
        return best_sim["code"], best_sim["score"], "Similarity ضعيف"




