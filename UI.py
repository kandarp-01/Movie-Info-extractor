
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="CineExtract AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #10131c 0%, #191c2b 100%);
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }

    .hero p {
        font-size: 1.1rem;
        color: #aeb5c7;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #ffffff;
    }

    .feature-card {
        background: #202536;
        border: 1px solid #343a50;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        min-height: 105px;
    }

    .feature-card h3 {
        margin: 0;
        font-size: 1.05rem;
        color: #ffffff;
    }

    .feature-card p {
        color: #aeb5c7;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #202536;
        color: #ffffff;
        border: 1px solid #424a63;
        border-radius: 12px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 10px;
        font-weight: 700;
        border: none;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7357ff, #a855f7);
        color: white;
    }

    div.stButton > button:hover {
        opacity: 0.88;
        border: 1px solid #a855f7;
    }

    .result-box {
        background: #202536;
        border: 1px solid #343a50;
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1rem;
    }

    .footer {
        text-align: center;
        color: #858da3;
        font-size: 0.85rem;
        padding-top: 2rem;
    }

    hr {
        border-color: #343a50;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- PROMPT TEMPLATE --------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intelligent movie information extraction assistant.

Analyze the provided paragraph and extract useful movie information.

Extract these details:
- Movie Name
- Release Year
- Director
- Genre
- Cast (main actors mentioned)
- Plot
- Setting
- Music Composer
- IMDb Rating
- Key Themes
- Quick Summary (2–4 sentences explaining the main premise
  and central conflict)

Instructions:
1. Use only information supported by the paragraph.
2. If information is unavailable, write "Not mentioned".
3. Never invent facts, cast members, ratings, or plot details.
4. Use clear headings and bullet points.
5. Keep the output concise and avoid repetition.
6. Make the summary easy to understand.
7. Return readable Markdown text, not JSON.
8. Keep extracted facts separate from the generated summary."""
    ),
    (
        "human",
        "Extract the movie information from this paragraph:\n\n{paragraph}"
    )
])

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

# -------------------- HEADER --------------------

st.markdown("""
<div class="hero">
    <h1>🎬 CineExtract AI</h1>
    <p>Turn movie descriptions into structured insights in seconds.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- FEATURES --------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🎥 Movie Details</h3>
        <p>Title, director, genre and cast</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Key Insights</h3>
        <p>Plot, setting and important themes</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>✨ Quick Summary</h3>
        <p>Understand the movie at a glance</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------- INPUT SECTION --------------------

st.markdown(
    '<div class="section-title">📝 Enter Your Movie Paragraph</div>',
    unsafe_allow_html=True
)

sample_paragraph = (
    "Interstellar is a visually stunning science fiction epic directed "
    "by Christopher Nolan. Released in 2014, the film stars Matthew "
    "McConaughey, Anne Hathaway, Jessica Chastain and Michael Caine. "
    "The story revolves around a group of astronauts who travel through "
    "a wormhole near Saturn in search of a new home for humanity as "
    "Earth faces environmental challenges. Hans Zimmer composed the "
    "powerful soundtrack. The film has an IMDb rating of 8.6."
)

if "movie_paragraph" not in st.session_state:
    st.session_state.movie_paragraph = ""

paragraph = st.text_area(
    "Paste the movie description below",
    key="movie_paragraph",
    height=220,
    placeholder=(
        "Paste a paragraph containing movie information here..."
    ),
    help="Enter a movie description, article excerpt, or review."
)

# -------------------- ACTION BUTTONS --------------------

col1, col2 = st.columns([3, 1])

with col1:
    extract_clicked = st.button(
        "✨ Extract Movie Information",
        type="primary",
        use_container_width=True
    )

with col2:
    sample_clicked = st.button(
        "Load Sample",
        use_container_width=True
    )

if sample_clicked:
    st.session_state.movie_paragraph = sample_paragraph
    st.rerun()

# -------------------- PROCESS INPUT --------------------

if extract_clicked:
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph first.")

    else:
        try:
            with st.spinner(
                "🎬 Analyzing the paragraph and extracting movie details..."
            ):
                model = load_model()

                chain = prompt | model
                response = chain.invoke({
                    "paragraph": paragraph.strip()
                })

                result = response.content

                st.session_state.movie_result = result

            st.success("Movie information extracted successfully!")

        except Exception as e:
            st.error(
                "Unable to process your paragraph. "
                "Please check your API key, model configuration, "
                "and internet connection."
            )
            with st.expander("View error details"):
                st.code(str(e))

# -------------------- RESULTS SECTION --------------------

if st.session_state.get("movie_result"):
    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Extracted Movie Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(st.session_state.movie_result)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Results",
            data=st.session_state.movie_result,
            file_name="movie_analysis.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        if st.button(
            "🗑️ Clear Results",
            use_container_width=True
        ):
            st.session_state.pop("movie_result", None)
            st.rerun()

# -------------------- FOOTER --------------------

st.markdown("""
<div class="footer">
    CineExtract AI · Powered by LangChain and Groq
</div>
""", unsafe_allow_html=True)
