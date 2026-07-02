"""
Nykaa Annual Report Chatbot
----------------------------
A Streamlit app styled to feel like the Nykaa app, that answers questions
about Nykaa's 2024-25 Annual Report using retrieval + Groq's free LLM API.

Deploy note: this uses TF-IDF retrieval (scikit-learn) instead of
sentence-transformers/chromadb so it installs fast and stays comfortably
inside Streamlit Community Cloud's free 1GB RAM limit.
"""

import os
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ----------------------------------------------------------------------
# Page config + Nykaa-style theming
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Nykaa Annual Report Assistant",
    page_icon="💄",
    layout="centered",
)

NYKAA_PINK = "#FC2779"
NYKAA_DARK = "#2D2926"
NYKAA_LIGHT_PINK = "#FFEAF3"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #FFF8FA;
    }}
    #MainMenu, footer {{visibility: hidden;}}

    .nykaa-header {{
        background: linear-gradient(135deg, {NYKAA_PINK} 0%, #FF6FA5 100%);
        padding: 22px 24px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(252, 39, 121, 0.25);
    }}
    .nykaa-header h1 {{
        color: white;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }}
    .nykaa-header p {{
        color: #FFE3EE;
        margin: 4px 0 0 0;
        font-size: 14px;
    }}

    .chip {{
        display: inline-block;
        background: white;
        color: {NYKAA_PINK};
        border: 1px solid {NYKAA_PINK};
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-size: 13px;
        font-weight: 600;
    }}

    [data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 4px 2px;
    }}

    .stButton>button {{
        background-color: {NYKAA_PINK};
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: 600;
        padding: 6px 18px;
    }}
    .stButton>button:hover {{
        background-color: #E01D68;
        color: white;
    }}

    .source-tag {{
        display: inline-block;
        background: {NYKAA_LIGHT_PINK};
        color: {NYKAA_PINK};
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 12px;
        margin-top: 6px;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nykaa-header">
        <h1>💄 Nykaa Report Assistant</h1>
        <p>Ask me anything about Nykaa's FY 2024–25 Annual Report — revenue, growth, leadership, employees, and more.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PDF_PATH = "Nykaa_Integrated-Annual-Report-2024-25.pdf"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

SUGGESTED_QUESTIONS = [
    "What are the key financial highlights?",
    "Who is the CEO of Nykaa?",
    "What is the growth strategy?",
    "How many employees work at Nykaa?",
    "Who is the brand ambassador?",
]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error(
            "No GROQ_API_KEY found. Add it under Settings → Secrets on Streamlit "
            "Community Cloud (see deployment instructions)."
        )
        st.stop()
    return key


@st.cache_resource(show_spinner="Reading the annual report and building the search index...")
def build_index(pdf_path: str):
    if not os.path.exists(pdf_path):
        st.error(f"Could not find '{pdf_path}'. Make sure the PDF is committed to the repo root.")
        st.stop()

    reader = PdfReader(pdf_path)
    chunks = []
    chunk_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        # simple sliding-window chunking per page
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
                chunk_pages.append(page_num)
            start += CHUNK_SIZE - CHUNK_OVERLAP

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(chunks)

    return {
        "chunks": chunks,
        "pages": chunk_pages,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def retrieve(index, question: str, k: int = TOP_K):
    q_vec = index["vectorizer"].transform([question])
    sims = cosine_similarity(q_vec, index["matrix"]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    top_idx = [i for i in top_idx if sims[i] > 0]
    context_chunks = [index["chunks"][i] for i in top_idx]
    pages = sorted({index["pages"][i] for i in top_idx})
    return context_chunks, pages


PROMPT_TEMPLATE = """You are a precise financial analyst answering questions about Nykaa's annual report.

Use ONLY the context below. Answer in 3-5 short bullet points, each one line, no sub-bullets.
State any numbers/percentages/names first, plainly, with no preamble like "Based on the context".
Do not repeat a point. Do not add generic commentary or disclaimers.
If the answer isn't in the context, reply exactly: "This information is not available in the report."

<CONTEXT>
{context}
</CONTEXT>

Question: {question}

Answer (max 5 bullet points, max ~80 words total):"""


def ask_groq(client, question: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


# ----------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------
api_key = get_api_key()
client = Groq(api_key=api_key)
index = build_index(PDF_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("**Quick questions:**")
cols = st.columns(len(SUGGESTED_QUESTIONS[:3]))
cols2 = st.columns(len(SUGGESTED_QUESTIONS[3:]))
clicked_question = None
for col, q in zip(cols, SUGGESTED_QUESTIONS[:3]):
    if col.button(q, use_container_width=True):
        clicked_question = q
for col, q in zip(cols2, SUGGESTED_QUESTIONS[3:]):
    if col.button(q, use_container_width=True):
        clicked_question = q

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="💄" if msg["role"] == "assistant" else "🛍️"):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.markdown(
                f'<span class="source-tag">📄 Source pages: {msg["pages"]}</span>',
                unsafe_allow_html=True,
            )

user_input = st.chat_input("Ask about Nykaa's revenue, growth, leadership...")
question = clicked_question or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🛍️"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="💄"):
        with st.spinner("Checking the annual report..."):
            context_chunks, pages = retrieve(index, question)
            answer = ask_groq(client, question, context_chunks)
        st.markdown(answer)
        if pages:
            st.markdown(
                f'<span class="source-tag">📄 Source pages: {pages}</span>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "pages": pages}
    )
