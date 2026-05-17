import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

st.set_page_config(
    page_title="Resume Screener AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.score-box {
    background: #f0f2f6;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.score-num { font-size: 3rem; font-weight: 800; line-height: 1; }
.score-lbl { font-size: 0.8rem; color: #666; margin-top: 0.3rem; }
.rank-card {
    background: #f8f9fb;
    border: 1px solid #e0e4ee;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rank-medal { font-size: 1.4rem; min-width: 2rem; }
.rank-name { font-weight: 600; font-size: 0.9rem; }
.rank-preview { font-size: 0.78rem; color: #777; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-pct { font-size: 1.1rem; font-weight: 700; min-width: 4rem; text-align: right; }
.verdict-yes { color: #16a34a; font-size: 0.72rem; font-weight: 600; }
.verdict-no  { color: #dc2626; font-size: 0.72rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Backend ─────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","if","while","is","am","are","was","were",
    "be","been","being","to","of","in","for","on","with","as","by","at","from",
    "this","that","these","those","it","its","into","about","over","after","before",
    "between","during","without","within","through","can","could","should","would",
    "may","might","will","just","also","we","you","they","he","she","i","me",
    "my","our","your","their","them","us","who","whom","which","what","where",
    "when","why","how","not","no","yes"
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t) > 1)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        st.error(f"PDF error: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        st.error(f"DOCX error: {e}")
        return ""


def extract_resume_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    st.error("Only PDF or DOCX supported.")
    return ""


def build_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "resume_text": "Computer Science student skilled in Python, SQL, pandas, NumPy, machine learning, data analysis, and web development. Experience with Google Colab, MySQL, APIs, GitHub, and problem solving.",
            "job_description": "Looking for a Python developer with machine learning, SQL, data analysis, pandas, NumPy, and web development skills.",
            "label": 1,
        },
        {
            "resume_text": "Computer Science pre-final year student with Python, SQL, web development, Shopify development, LinkedIn Sales Navigator outreach, and data labeling experience.",
            "job_description": "Hiring a technical intern with Python, SQL, web development, and communication skills.",
            "label": 1,
        },
        {
            "resume_text": "Graphic designer with Photoshop, Illustrator, branding, logo design, and animation skills.",
            "job_description": "Hiring a backend Python developer with APIs and machine learning experience.",
            "label": 0,
        },
    ])


@st.cache_resource(show_spinner="Training model…")
def get_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from xgboost import XGBClassifier

    df = build_demo_dataset()
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    df["job_description"] = df["job_description"].fillna("").astype(str)

    cleaner = FunctionTransformer(lambda d: d.apply(lambda col: col.map(clean_text)), validate=False)
    features = ColumnTransformer([
        ("r", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "resume_text"),
        ("j", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "job_description"),
    ])
    model = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )
    pipe = Pipeline([("cleaner", cleaner), ("features", features), ("classifier", model)])
    pipe.fit(df[["resume_text", "job_description"]], df["label"])
    return pipe


def score_resume(pipeline, resume_text: str, job_desc: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rc = clean_text(resume_text)
    jc = clean_text(job_desc)

    tv = TfidfVectorizer(ngram_range=(1, 2))
    mat = tv.fit_transform([rc, jc])
    cos = float(cosine_similarity(mat[0], mat[1])[0][0])

    rw, jw = set(rc.split()), set(jc.split())
    overlap = len(rw & jw) / max(len(jw), 1)

    skills = {"python","sql","pandas","numpy","mysql","web","development","data",
              "analysis","machine","learning","colab","github","api","apis"}
    bonus = len(rw & skills) * 0.05

    final = min(0.60 * cos + 0.30 * overlap + bonus, 1.0)
    return (1 if final >= 0.35 else 0), final


def rank_resumes(pipeline, resumes: List[str], names: List[str], job_desc: str) -> pd.DataFrame:
    rows = []
    for r, n in zip(resumes, names):
        pred, score = score_resume(pipeline, r, job_desc)
        rows.append({"name": n, "resume_text": r,
                     "match_score": round(score * 100, 1), "prediction": pred})
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)


def score_color(s): return "#16a34a" if s >= 70 else "#d97706" if s >= 40 else "#dc2626"


# ── UI ──────────────────────────────────────────────────────

st.title("🎯 Resume Screener AI")
st.caption("TF-IDF · XGBoost · Cosine Similarity — paste or upload résumés to get match scores instantly.")

pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.header("📋 Job Description")
    job_description = st.text_area(
        "Job Description",
        value="Looking for a Python developer / data analyst intern with Python, SQL, pandas, NumPy, data analysis, web development, MySQL, Google Colab, and problem-solving skills.",
        height=250,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.info("Paste the job description above, then use any tab to analyse résumés.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Resume", "📊 Batch Ranking", "📁 Custom Dataset"])


# ── Tab 1: Single ───────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("Resume Input")
        method = st.radio("Input method", ["Upload PDF / DOCX", "Paste text"], horizontal=True)
        resume_text_single = ""

        if method == "Upload PDF / DOCX":
            f = st.file_uploader("Upload", type=["pdf","docx"], label_visibility="collapsed")
            if f:
                resume_text_single = extract_resume_text(f)
                if resume_text_single:
                    with st.expander("Preview extracted text"):
                        st.text(resume_text_single[:1500])
        else:
            resume_text_single = st.text_area("Resume text", height=280,
                                               placeholder="Paste full résumé here…",
                                               label_visibility="collapsed")

    with col2:
        st.subheader("Result")
        if st.button("Analyse ▶", use_container_width=True, type="primary"):
            if not resume_text_single.strip():
                st.warning("Please provide a résumé.")
            elif not job_description.strip():
                st.warning("Add a job description in the sidebar.")
            else:
                with st.spinner("Scoring…"):
                    pred, score = score_resume(pipeline, resume_text_single, job_description)
                pct = round(score * 100, 1)
                color = score_color(pct)

                st.markdown(f"""
                <div class="score-box">
                  <div class="score-num" style="color:{color}">{pct}%</div>
                  <div class="score-lbl">Match Score</div>
                </div>
                """, unsafe_allow_html=True)

                if pred == 1:
                    st.success("✅ SHORTLISTED")
                else:
                    st.error("❌ REJECTED")

                st.divider()

                # Breakdown
                rc = clean_text(resume_text_single)
                jc = clean_text(job_description)
                from sklearn.feature_extraction.text import TfidfVectorizer as TV
                from sklearn.metrics.pairwise import cosine_similarity as CS
                tv = TV(ngram_range=(1, 2))
                mat = tv.fit_transform([rc, jc])
                cos_pct = round(float(CS(mat[0], mat[1])[0][0]) * 100, 1)
                rw, jw = set(rc.split()), set(jc.split())
                ov_pct = round(len(rw & jw) / max(len(jw), 1) * 100, 1)

                c1, c2 = st.columns(2)
                c1.metric("Cosine Similarity", f"{cos_pct}%")
                c2.metric("Keyword Overlap", f"{ov_pct}%")

                skills = {"python","sql","pandas","numpy","mysql","web","development","data",
                          "analysis","machine","learning","colab","github","api","apis"}
                hits = sorted(rw & skills)
                if hits:
                    st.markdown("**Matched Skills:**")
                    st.write("  •  ".join(hits))


# ── Tab 2: Batch ────────────────────────────────────────────
with tab2:
    st.subheader("Batch Resume Ranking")
    method2 = st.radio("Batch input", ["Upload multiple files", "Paste multiple texts"], horizontal=True)

    batch_texts, batch_names = [], []

    if method2 == "Upload multiple files":
        files = st.file_uploader("Upload résumés", type=["pdf","docx"],
                                  accept_multiple_files=True, label_visibility="collapsed")
        if files:
            for f in files:
                t = extract_resume_text(f)
                if t:
                    batch_texts.append(t)
                    batch_names.append(f.name)
    else:
        n = st.number_input("Number of résumés", 2, 10, 3, step=1)
        for i in range(int(n)):
            t = st.text_area(f"Resume {i+1}", height=110, key=f"b{i}",
                             placeholder=f"Paste résumé #{i+1}…")
            batch_texts.append(t)
            batch_names.append(f"Resume #{i+1}")

    if st.button("Rank All ▶", use_container_width=True, type="primary"):
        valid = [(t, n) for t, n in zip(batch_texts, batch_names) if t.strip()]
        if not valid:
            st.warning("Add at least one résumé.")
        elif not job_description.strip():
            st.warning("Add a job description in the sidebar.")
        else:
            with st.spinner("Ranking…"):
                texts = [v[0] for v in valid]
                names = [v[1] for v in valid]
                ranked = rank_resumes(pipeline, texts, names, job_description)

            st.divider()
            medals = ["🥇","🥈","🥉"] + [f"#{i+1}" for i in range(3, 20)]

            for i, row in ranked.iterrows():
                color = score_color(row["match_score"])
                verdict_cls = "verdict-yes" if row["prediction"] == 1 else "verdict-no"
                verdict_txt = "✓ Shortlist" if row["prediction"] == 1 else "✗ Reject"
                preview = row["resume_text"][:80].replace("\n"," ")

                st.markdown(f"""
                <div class="rank-card">
                  <div class="rank-medal">{medals[i]}</div>
                  <div style="flex:1;min-width:0">
                    <div class="rank-name">{row['name']}</div>
                    <div class="rank-preview">{preview}…</div>
                  </div>
                  <div style="text-align:right">
                    <div class="rank-pct" style="color:{color}">{row['match_score']}%</div>
                    <div class="{verdict_cls}">{verdict_txt}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            csv = ranked[["name","match_score","prediction"]].copy()
            csv["verdict"] = csv["prediction"].map({1:"Shortlist",0:"Reject"})
            csv = csv.drop(columns=["prediction"])
            st.download_button("⬇ Download CSV", csv.to_csv(index=False),
                               "ranked_resumes.csv", "text/csv")


# ── Tab 3: Custom Dataset ───────────────────────────────────
with tab3:
    st.subheader("Train on Custom Dataset")
    st.caption("CSV must have columns: `resume_text`, `job_description`, `label` (1=match, 0=no match)")

    with st.expander("View demo dataset"):
        st.dataframe(build_demo_dataset(), use_container_width=True)

    csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if csv_file:
        try:
            df_c = pd.read_csv(csv_file)
            missing = {"resume_text","job_description","label"} - set(df_c.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            else:
                st.success(f"Loaded {len(df_c)} rows.")
                st.dataframe(df_c.head(), use_container_width=True)

                if st.button("Train Model ▶", type="primary"):
                    from sklearn.compose import ColumnTransformer
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.model_selection import train_test_split
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import FunctionTransformer
                    from sklearn.metrics import accuracy_score, classification_report
                    from xgboost import XGBClassifier

                    df_c["resume_text"] = df_c["resume_text"].fillna("").astype(str)
                    df_c["job_description"] = df_c["job_description"].fillna("").astype(str)
                    df_c["label"] = df_c["label"].astype(int)
                    X = df_c[["resume_text","job_description"]]
                    y = df_c["label"]
                    strat = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

                    cleaner = FunctionTransformer(lambda d: d.apply(lambda col: col.map(clean_text)), validate=False)
                    features = ColumnTransformer([
                        ("r", TfidfVectorizer(max_features=3000, ngram_range=(1,2)), "resume_text"),
                        ("j", TfidfVectorizer(max_features=3000, ngram_range=(1,2)), "job_description"),
                    ])
                    mdl = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                        random_state=42, eval_metric="logloss", n_jobs=-1)
                    p = Pipeline([("cleaner",cleaner),("features",features),("classifier",mdl)])

                    with st.spinner("Training…"):
                        p.fit(X_tr, y_tr)
                        acc = accuracy_score(y_te, p.predict(X_te))
                        rep = classification_report(y_te, p.predict(X_te), zero_division=0)

                    st.success(f"Done! Test Accuracy: **{acc:.2%}**")
                    st.code(rep)
        except Exception as e:
            st.error(f"Error: {e}")