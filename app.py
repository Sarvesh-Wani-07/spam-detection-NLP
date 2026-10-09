import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RS = 42
HAM, SPAM = "#3FB6A8", "#F2A33A"
DATA_PATH = Path(__file__).parent / "SMSSpamCollection"

st.set_page_config(page_title="SMS Spam Lab", page_icon="📨", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
.verdict{border-radius:12px;padding:1.2rem 1.5rem;font-size:1.9rem;font-weight:700}
.verdict small{display:block;font-size:.9rem;font-weight:400;opacity:.85;margin-top:.2rem}
.spam{background:#F2A33A22;border:1px solid #F2A33A;color:#F2A33A}
.ham{background:#3FB6A822;border:1px solid #3FB6A8;color:#3FB6A8}
</style>""", unsafe_allow_html=True)

EXAMPLES = {
    "Prize scam": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt WORD to 81010",
    "Casual chat": "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "Fake delivery": "Your parcel is held at the depot. Pay the £1.99 fee at www.track-parcel.co now to reschedule.",
    "Tricky ham": "Can you call me when you get a sec? Won't take long, promise. Don't forget the free tickets!",
}


# ---------- pipeline (same logic as the notebook) ----------
@st.cache_resource
def setup_nltk():
    for p in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        nltk.download(p, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


STOP, LEMMA = setup_nltk()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"£|\$", " CURRENCY ", text)
    text = re.sub(r"\b\d{4,}\b", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(LEMMA.lemmatize(t) for t in word_tokenize(text) if t not in STOP and len(t) > 1)


@st.cache_resource(show_spinner="Training four models…")
def train(raw: pd.DataFrame):
    df = raw.drop_duplicates(subset=["message"]).reset_index(drop=True)
    df["spam"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["message"].apply(clean_text)
    df["chars"] = df["message"].str.len()
    df["words"] = df["message"].str.split().apply(len)

    Xtr_t, Xte_t, ytr, yte = train_test_split(df["clean"], df["spam"], test_size=0.2,
                                              stratify=df["spam"], random_state=RS)
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    Xtr, Xte = tfidf.fit_transform(Xtr_t), tfidf.transform(Xte_t)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RS)),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RS, n_jobs=-1),
    }
    out = {"df": df, "tfidf": tfidf, "models": {}, "yte": yte.values, "proba": {}}
    rows = []
    for n, m in models.items():
        m.fit(Xtr, ytr)
        out["models"][n] = m
        p = m.predict_proba(Xte)[:, 1]
        out["proba"][n] = p
        yp = (p >= 0.5).astype(int)
        rows.append({"Model": n, "Accuracy": accuracy_score(yte, yp), "Precision": precision_score(yte, yp),
                     "Recall": recall_score(yte, yp), "F1-score": f1_score(yte, yp), "ROC-AUC": roc_auc_score(yte, p)})
    out["results"] = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    out["n_dupes"] = len(raw) - len(df)
    return out


def load_raw():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    up = st.file_uploader("Upload `SMSSpamCollection` (tab-separated: label, message)")
    if up is None:
        st.info("Place `SMSSpamCollection` next to `app.py`, or upload it above.")
        st.stop()
    return pd.read_csv(up, sep="\t", header=None, names=["label", "message"])


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
    return fig


# ---------- load ----------
S = train(load_raw())
df, tfidf, models, results = S["df"], S["tfidf"], S["models"], S["results"]
feat = np.array(tfidf.get_feature_names_out())
best = results.iloc[0]["Model"]

with st.sidebar:
    st.title("📨 SMS Spam Lab")
    st.caption("TF-IDF + classical ML on the UCI SMS Spam Collection")
    model_name = st.selectbox("Model", list(models), index=list(models).index(best),
                              help=f"Best by ROC-AUC: {best}")
    thr = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05,
                    help="Lower = catches more spam but flags more real messages.")
    st.divider()
    st.metric("Messages (deduplicated)", f"{len(df):,}")
    st.metric("Duplicates removed", S["n_dupes"])
    st.metric("Spam share", f"{df['spam'].mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live classifier", "📊 Model performance", "🗂 Data explorer", "🧠 What it learned"])

# ---------- tab 1 ----------
with tab1:
    st.subheader("Try a message")
    cols = st.columns(len(EXAMPLES))
    for c, (k, v) in zip(cols, EXAMPLES.items()):
        if c.button(k, width="stretch"):
            st.session_state["msg"] = v
    msg = st.text_area("Message", key="msg", height=110, placeholder="Type or paste an SMS…",
                       label_visibility="collapsed")

    if msg.strip():
        clean = clean_text(msg)
        vec = tfidf.transform([clean])
        p = models[model_name].predict_proba(vec)[0, 1]
        is_spam = p >= thr
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"<div class='verdict {'spam' if is_spam else 'ham'}'>"
                        f"{'🚨 SPAM' if is_spam else '✅ HAM'}<small>spam probability {p:.1%} · "
                        f"threshold {thr:.0%} · {model_name}</small></div>", unsafe_allow_html=True)
            st.caption("**After cleaning:** " + (clean or "*(nothing left after stopword removal)*"))
        with right:
            g = go.Figure(go.Indicator(mode="gauge", value=p * 100,
                          gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=SPAM if is_spam else HAM),
                                     threshold=dict(line=dict(color="white", width=3), value=thr * 100))))
            st.plotly_chart(style(g, 170), width="stretch")

        st.markdown("##### Why? Words that moved the score")
        lr = models["Logistic Regression"]
        idx = vec.nonzero()[1]
        if len(idx):
            contrib = pd.DataFrame({"term": feat[idx], "impact": lr.coef_[0][idx] * vec[0, idx].toarray()[0]})
            contrib = contrib.reindex(contrib.impact.abs().sort_values(ascending=False).index).head(12)
            contrib["side"] = np.where(contrib.impact > 0, "Pushes to spam", "Pushes to ham")
            f = px.bar(contrib.iloc[::-1], x="impact", y="term", color="side", orientation="h",
                       color_discrete_map={"Pushes to spam": SPAM, "Pushes to ham": HAM})
            st.plotly_chart(style(f, 80 + 28 * len(contrib)), width="stretch")
            st.caption("Explanation uses Logistic Regression coefficients × TF-IDF weight (a transparent proxy for the other models).")
        else:
            st.info("None of the message's words are in the model's vocabulary.")

        st.markdown("##### All models, same message")
        cmp = pd.DataFrame({n: [m.predict_proba(vec)[0, 1]] for n, m in models.items()}).T
        cmp.columns = ["Spam probability"]
        f = px.bar(cmp, x="Spam probability", y=cmp.index, orientation="h", range_x=[0, 1])
        f.update_traces(marker_color=SPAM)
        st.plotly_chart(style(f, 220), width="stretch")
    else:
        st.info("Pick an example or type a message to classify it.")

    with st.expander("📥 Batch classify (CSV with a `message` column)"):
        up = st.file_uploader("CSV", type="csv", key="batch")
        if up:
            b = pd.read_csv(up)
            if "message" in b:
                b["spam_probability"] = models[model_name].predict_proba(tfidf.transform(b["message"].apply(clean_text)))[:, 1]
                b["prediction"] = np.where(b.spam_probability >= thr, "spam", "ham")
                st.dataframe(b, width="stretch")
                st.download_button("Download results", b.to_csv(index=False), "predictions.csv")
            else:
                st.error("CSV needs a `message` column.")

# ---------- tab 2 ----------
with tab2:
    top = results.iloc[0]
    c = st.columns(5)
    for col, m in zip(c, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
        col.metric(m, f"{top[m]:.3f}")
    st.caption(f"Showing **{best}** (best ROC-AUC) on the held-out 20% test set.")
    st.dataframe(results.style.format({k: "{:.3f}" for k in results.columns[1:]})
                 .background_gradient(cmap="YlOrBr", subset=list(results.columns[1:])), width="stretch", hide_index=True)
    melt = results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(style(px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", range_y=[0.6, 1.0],
                                 title="Model comparison (y-axis starts at 0.6)")), width="stretch")

    a, b = st.columns(2)
    yte = S["yte"]
    with a:
        r = go.Figure()
        for n, p in S["proba"].items():
            fpr, tpr, _ = roc_curve(yte, p)
            r.add_scatter(x=fpr, y=tpr, name=n, mode="lines")
        r.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Chance")
        r.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(r), width="stretch")
    with b:
        pr = go.Figure()
        for n, p in S["proba"].items():
            pre, rec, _ = precision_recall_curve(yte, p)
            pr.add_scatter(x=rec, y=pre, name=n, mode="lines")
        pr.update_layout(title="Precision–recall curves", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(style(pr), width="stretch")

    cm_model = st.radio("Confusion matrix for", list(models), index=list(models).index(best), horizontal=True)
    cm = confusion_matrix(yte, (S["proba"][cm_model] >= thr).astype(int))
    h = px.imshow(cm, text_auto=True, x=["Ham", "Spam"], y=["Ham", "Spam"], color_continuous_scale="YlOrBr",
                  labels=dict(x="Predicted", y="Actual"), title=f"{cm_model} @ threshold {thr:.2f}")
    st.plotly_chart(style(h, 340), width="stretch")

# ---------- tab 3 ----------
with tab3:
    a, b = st.columns(2)
    cc = df["label"].value_counts()
    d = px.pie(values=cc.values, names=cc.index, hole=.55, color=cc.index, title="Class balance",
               color_discrete_map={"ham": HAM, "spam": SPAM})
    a.plotly_chart(style(d), width="stretch")
    h = px.histogram(df, x="chars", color="label", barmode="overlay", opacity=.65, nbins=60, range_x=[0, 400],
                     histnorm="probability density", title="Message length (characters)",
                     color_discrete_map={"ham": HAM, "spam": SPAM})
    b.plotly_chart(style(h), width="stretch")

    def topw(flag, n=15):
        return pd.DataFrame(Counter(" ".join(df[df.spam == flag]["clean"]).split()).most_common(n), columns=["word", "count"])
    a, b = st.columns(2)
    a.plotly_chart(style(px.bar(topw(1).iloc[::-1], x="count", y="word", orientation="h", title="Top words — spam",
                                color_discrete_sequence=[SPAM])), width="stretch")
    b.plotly_chart(style(px.bar(topw(0).iloc[::-1], x="count", y="word", orientation="h", title="Top words — ham",
                                color_discrete_sequence=[HAM])), width="stretch")
    st.markdown("##### Browse messages")
    f1, f2 = st.columns([1, 3])
    lab = f1.selectbox("Label", ["all", "spam", "ham"])
    q = f2.text_input("Search text")
    v = df if lab == "all" else df[df.label == lab]
    if q:
        v = v[v.message.str.contains(q, case=False, regex=False)]
    st.dataframe(v[["label", "message", "clean"]].head(500), width="stretch", hide_index=True)

# ---------- tab 4 ----------
with tab4:
    coefs = models["Logistic Regression"].coef_[0]
    n = st.slider("Words to show", 10, 30, 20)
    sp, hm = np.argsort(coefs)[-n:], np.argsort(coefs)[:n]
    a, b = st.columns(2)
    a.plotly_chart(style(px.bar(x=coefs[sp], y=feat[sp], orientation="h", title="Strongest spam signals",
                                color_discrete_sequence=[SPAM]), 80 + 22 * n), width="stretch")
    b.plotly_chart(style(px.bar(x=coefs[hm][::-1], y=feat[hm][::-1], orientation="h", title="Strongest ham signals",
                                color_discrete_sequence=[HAM]), 80 + 22 * n), width="stretch")
    st.caption("Logistic Regression coefficients on TF-IDF features (unigrams + bigrams, top 3,000).")
