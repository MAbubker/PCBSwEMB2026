import streamlit as st
import pandas as pd
import io
import json

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

from utils import (
    clean_text,
    extract_keywords,
    hybrid_predict,
    predict_similarity_from_embedding,
    save_feedback,
    model
)

# ==========================================
# إعداد الصفحة
# ==========================================
st.set_page_config(page_title="نظام الترميز الذكي", layout="wide")

st.title("🧠 نظام الترميز الاقتصادي الذكي (ISIC-4)")
st.subheader("الجهاز المركزي للاحصاء الفلسطيني")
st.subheader("**ملاحظة: ضرورة وجود عمودين في ملف الاكسل المحمل: activity_description و establishment_name **")

# ==========================================
# إعدادات
# ==========================================
num_suggestions = st.slider("عدد الاقتراحات", 3, 15, 5)

uploaded_file = st.file_uploader("📂 تحميل ملف Excel", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

if "activity_description" not in df.columns:
    st.error("❌ يجب وجود عمود activity_description")
    st.stop()

# ==========================================
# تجهيز البيانات
# ==========================================
texts = []

for _, row in df.iterrows():
    activity = clean_text(row["activity_description"])
    est = clean_text(row.get("establishment_name", ""))
    keywords = extract_keywords(est)

    full_text = activity if est == "" else f"{activity} {activity} {activity} | {keywords} | {est}"
    texts.append(full_text)

with st.spinner("⏳ جاري التحليل..."):
    all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# ==========================================
# بناء البيانات
# ==========================================
table_data = []

for i, row in df.iterrows():

    emb = all_embeddings[i]

    activity = row["activity_description"]
    est = row.get("establishment_name", "")

    pred_code, score, source = hybrid_predict(texts[i], emb, num_suggestions)
    top_matches = predict_similarity_from_embedding(emb, num_suggestions)

    similarity_score = round(top_matches.iloc[0]["score"], 2)

    desc_row = top_matches[top_matches["code"] == pred_code]
    desc = desc_row.iloc[0]["description"] if not desc_row.empty else ""
   
    options = list([
    f"{r['code']} - {r['description']} ({round(r['score'],2)})"
    for _, r in top_matches.head(num_suggestions).iterrows()
    ])


    table_data.append({
        "اسم المنشأة": est,
        "النشاط": activity,
        "التشابه": similarity_score,
        "الكود المقترح": pred_code,
        "الوصف": desc,
        "اختيار المستخدم": options[0] if options else "",
        "options": options,
        "كود يدوي": "",
        "نسبة الثقة": source
    })

df_table = pd.DataFrame(table_data)

df_table["options"] = df_table["options"].apply(
    lambda x: x if isinstance(x, list) else []
)

# ==========================================
# AG-GRID إعداد
# ==========================================
st.markdown("## 🧾 شاشة الترميز (ISIC-4)")

gb = GridOptionsBuilder.from_dataframe(df_table)

# جعل الأعمدة قابلة للتصفية والفرز
gb.configure_default_column(
    editable=True,
    filter=True,
    sortable=True,
    resizable=True
)

# ==========================================
# Dropdown لكل صف 🔥🔥🔥
# ==========================================
cell_editor = JsCode("""
class DropdownEditor {
    init(params) {
        this.params = params;

        this.eInput = document.createElement('select');
        this.eInput.style.width = '100%';

        let options = params.data.options || [];

        options.forEach(opt => {
            let option = document.createElement('option');
            option.value = opt;
            option.innerHTML = opt;
            this.eInput.appendChild(option);
        });

        this.eInput.value = params.value;
    }

    getGui() {
        return this.eInput;
    }

    afterGuiAttached() {
        this.eInput.focus();
    }

    getValue() {
        return this.eInput.value;
    }

    isPopup() {
        return true;
    }
}
""")



# عمود غير مهم للإظهار
gb.configure_column(
    "اختيار المستخدم",
    editable=True,
    cellEditor='agSelectCellEditor',
    cellEditorParams={
        "values": list(set(sum(df_table["options"], [])))
    }
)

grid_options = gb.build()

# ==========================================
# عرض الجدول
# ==========================================
grid_response = AgGrid(
    df_table,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
    height=500,
    reload_data=True
)

edited_df = pd.DataFrame(grid_response["data"])

# ==========================================
# الحفظ
# ==========================================
st.markdown("### 💾 الحفظ")

if st.button("💾 حفظ التعديلات"):

    for i, row in edited_df.iterrows():

        selected_code = ""

        if isinstance(row["اختيار المستخدم"], str) and " - " in row["اختيار المستخدم"]:
            selected_code = row["اختيار المستخدم"].split(" - ")[0]
        else:
            selected_code = str(row["اختيار المستخدم"])

        if str(row["كود يدوي"]).strip():
            selected_code = str(row["كود يدوي"]).strip()

        save_feedback(
            texts[i],
            selected_code,
            row["النشاط"]
        )

    st.success("✅ تم الحفظ والتعلم بنجاح")

# ==========================================
# تصدير
# ==========================================
st.markdown("### 📊 تصدير")

buffer = io.BytesIO()
edited_df.to_excel(buffer, index=False)
buffer.seek(0)

st.download_button(
    "📥 تحميل النتائج",
    buffer,
    "results.xlsx"
)