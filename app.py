import html

import streamlit as st

from retrieval import FaqSearcher, MIN_SCORE, load_faqs

st.set_page_config(
    page_title="CampusBuddy AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #10172a 0%, #172554 55%, #312e81 100%); }
    [data-testid="stSidebar"] { background: #111827; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 22px;
        background: linear-gradient(110deg, rgba(14,165,233,.24), rgba(139,92,246,.28));
        border: 1px solid rgba(255,255,255,.16); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.5rem; color: white; }
    .hero p { color: #dbeafe; margin-bottom: 0; font-size: 1.05rem; }
    .answer-card h4 { margin: 0 0 .5rem 0; color: #f8fafc; }
    .answer-card p { margin: .35rem 0; color: #e2e8f0; }
    .answer-card {
        padding: 1.2rem; border-radius: 18px; background: rgba(15,23,42,.82);
        border: 1px solid rgba(125,211,252,.3); margin-top: .7rem;
    }
    .muted { color: #cbd5e1; }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.65); padding: 14px; border-radius: 15px;
        border: 1px solid rgba(255,255,255,.12);
    }
    .stButton button { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_searcher():
    """Load and index the FAQs once per server run."""
    return FaqSearcher(load_faqs())


try:
    searcher = get_searcher()
    faqs = searcher.faqs
except ValueError as error:
    st.error(f"I couldn't load the FAQ data: {error}")
    st.stop()

with st.sidebar:
    st.markdown("## 🎓 CampusBuddy AI")
    st.caption("Your mini AI-powered campus guide")
    st.markdown("---")
    st.markdown("### 🧠 How it works")
    st.write("1. You type a question.")
    st.write("2. TF-IDF turns words into numbers (after expanding abbreviations like ML or NLP).")
    st.write("3. Cosine similarity finds the closest FAQ.")
    st.write("4. The app shows the best matching answer.")
    st.markdown("---")
    st.markdown("### ⚙️ Try changing it")
    st.caption("Edit `data/faqs.csv` to add your own campus questions and answers.")
    st.markdown("---")
    st.caption("Built with Python • Streamlit • scikit-learn")

st.markdown("""
<div class="hero">
    <h1>🎓 CampusBuddy AI</h1>
    <p>Your smart student helpdesk — ask a question and let a little NLP find the closest answer.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("📚 Knowledge base", f"{len(faqs)} FAQs")
col2.metric("🧠 AI technique", "TF-IDF + cosine similarity")
col3.metric("💻 App type", "NLP retrieval chatbot")

st.markdown("### 💬 Ask CampusBuddy")
st.write("Try a question below, or write your own.")

suggestions = [
    "How can I prepare for exams?",
    "What is machine learning?",
    "How do I manage my time?",
    "What should I include in a project README?",
]


def use_suggestion(text):
    """Button callback: fill the box and run the search straight away."""
    st.session_state["question_input"] = text
    st.session_state["run_search"] = True


scols = st.columns(4)
for i, suggestion in enumerate(suggestions):
    scols[i].button(suggestion, width="stretch",
                    on_click=use_suggestion, args=(suggestion,), key=f"suggestion_{i}")

question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="Example: How can I make a good study timetable?",
)

clicked = st.button("✨ Find my answer", type="primary", width="stretch")
run_search = clicked or st.session_state.pop("run_search", False)

if run_search:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        results = searcher.search(question.strip(), top_k=3)
        result, score = results[0]

        if score < MIN_SCORE:
            st.warning("I couldn't find a close match. Try using different words or add a related FAQ to data/faqs.csv.")
        else:
            # Escape CSV text so special characters can't break the page layout.
            st.markdown(
                f'''<div class="answer-card">
                <h4>💡 {html.escape(result["category"])}</h4>
                <p><b>Closest question:</b> {html.escape(result["question"])}</p>
                <p>{html.escape(result["answer"])}</p>
                </div>''',
                unsafe_allow_html=True,
            )
            left, right = st.columns([2, 1])
            with left:
                st.progress(min(score, 1.0), text=f"Text similarity score: {score * 100:.1f}%")
            with right:
                st.caption("This score measures text similarity, not guaranteed correctness.")

            related = [r["question"] for r, sc in results[1:] if sc >= MIN_SCORE]
            if related:
                st.markdown("**You might also ask:**")
                for text in related:
                    st.markdown(f"- {text}")

st.markdown("---")
with st.expander("🔍 Explore the knowledge base"):
    st.dataframe(faqs, width="stretch", hide_index=True)

with st.expander("🧪 Understand the AI"):
    st.markdown("""
    **TF-IDF** gives more importance to words that are useful for distinguishing one question from another.

    **Cosine similarity** compares the user's question with every saved FAQ question. The closest match is selected.

    **Important limitation:** This is a small retrieval chatbot, not a human and not a generative AI model. It can misunderstand unusual wording, and its similarity score is not a probability that an answer is correct.
    """)
