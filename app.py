import json
import os
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="CineSense | Movie Review Classifier",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("artifacts/sentiment_model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at top left, #30204d 0, #151526 42%, #0c0d16 100%);
  color: #f8fafc;
}
.block-container { max-width: 850px; padding-top: 2rem; }
.hero {
  padding: 1.5rem; border: 1px solid rgba(255,255,255,.13);
  border-radius: 22px; background: rgba(255,255,255,.06);
  margin-bottom: 1.2rem;
}
.eyebrow { color: #c4b5fd; font-size: .8rem; letter-spacing: .16em; font-weight: 700; }
.hero h1 { margin: .25rem 0; font-size: clamp(2rem, 7vw, 3.2rem); }
.hero p { color: #d1d5db; margin-bottom: 0; }
.result {
  padding: 1rem 1.2rem; border-radius: 16px; margin-top: 1rem;
  border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.07);
}
div.stButton > button {
  width: 100%; border-radius: 12px; border: 0; min-height: 3rem;
  background: #a78bfa; color: #151526; font-weight: 800;
}
div.stButton > button:hover { background: #c4b5fd; color: #151526; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">MACHINE LEARNING • SENTIMENT ANALYSIS</div>
  <h1>🎬 CineSense</h1>
  <p>Discover the sentiment behind a movie review. Enter a review below and let the ML model classify it as positive or negative.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("The trained model was not found. Run `python train_model.py` first to create artifacts/sentiment_model.joblib.")
    st.stop()

with st.form("review_form"):
    review = st.text_area(
        "✍️ Write your movie review",
        placeholder="Example: The story was engaging, the acting was excellent, and I loved every minute of it...",
        height=160,
        max_chars=10000,
    )
    submitted = st.form_submit_button("✨ Analyze Review")

if submitted:
    cleaned_review = review.strip()
    if len(cleaned_review.split()) < 3:
        st.warning("Please enter a little more detail (at least 3 words) for a more useful prediction.")
    else:
        prediction = model.predict([cleaned_review])[0]
        probabilities = model.predict_proba([cleaned_review])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])
        if prediction == "pos":
            st.markdown(f"""
            <div class="result">
              <h2>🟢 Positive Review</h2>
              <p>The model predicts a positive sentiment.</p>
              <h3>{confidence:.1%} model confidence</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result">
              <h2>🔴 Negative Review</h2>
              <p>The model predicts a negative sentiment.</p>
              <h3>{confidence:.1%} model confidence</h3>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Confidence is the model's estimated probability, not a guarantee that the prediction is correct.")

with st.expander("📊 Model details"):
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews used", f"{metrics.get('total_reviews', 0):,}")
        c2.metric("Training samples", f"{metrics.get('train_reviews', 0):,}")
        c3.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        st.write("Algorithm: TF-IDF features + Logistic Regression")
        st.write("Dataset: NLTK movie_reviews corpus (balanced positive and negative reviews).")
        st.caption("Accuracy is measured on a held-out test split. Results can vary with dataset and split.")
    else:
        st.info("Run train_model.py to generate evaluation metrics.")

st.divider()
st.caption("Student ML project • English-language reviews • Binary sentiment classification")
