
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="CineExtract AI",
    page_icon="🎬",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #10131c, #1b2030);
        color: #ffffff;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
    }

    .hero p {
        color: #aeb5c7;
        font-size: 1.1rem;
    }

    .info-card {
        background: #202536;
        border: 1px solid #343a50;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .info-label {
        color: #aeb5c7;
        font-size: 0.9rem;
    }

    .info-value {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .section-heading {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #202536;
        color: #ffffff;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7357ff, #a855f7);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- PYDANTIC MODEL --------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# -------------------- OUTPUT PARSER --------------------

parser = PydanticOutputParser(pydantic_object=Movie)


# -------------------- PROMPT TEMPLATE --------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract movie information from the paragraph.

Extract the movie title, release year, genre, director, cast,
rating, and a concise summary of the movie.

Rules:
1. Use only information supported by the paragraph.
2. Do not invent facts.
3. If the year, director, or rating is missing, use null.
4. If the genre or cast is missing, return an empty list.
5. If the paragraph does not contain enough information for a
   meaningful summary, clearly state that in the summary.
6. The rating must be a number when explicitly provided.
7. Follow the required output format exactly.

{format_instructions}
"""
    ),
    (
        "human",
        "Extract movie information from this paragraph:\n\n{paragraph}"
    )
])


# -------------------- MODEL --------------------

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
    <p>Extract structured movie information from any paragraph using AI.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- INPUT SECTION --------------------

st.markdown(
    '<div class="section-heading">📝 Movie Description</div>',
    unsafe_allow_html=True
)

sample_paragraph = """
Interstellar is a science fiction film directed by Christopher Nolan.
Released in 2014, it stars Matthew McConaughey, Anne Hathaway,
Jessica Chastain, and Michael Caine. The story follows a team of
astronauts who travel through a wormhole near Saturn in search of a
new home for humanity. Hans Zimmer composed the soundtrack.
The film has an IMDb rating of 8.6.
"""

if "paragraph" not in st.session_state:
    st.session_state.paragraph = ""

paragraph = st.text_area(
    "Paste your movie paragraph",
    key="paragraph",
    height=220,
    placeholder="Paste a movie description, article, or review here..."
)

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
    st.session_state.paragraph = sample_paragraph
    st.rerun()


# -------------------- EXTRACTION --------------------

if extract_clicked:
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph first.")

    else:
        try:
            with st.spinner("Analyzing the paragraph..."):
                model = load_model()

                final_prompt = prompt.invoke({
                    "format_instructions": parser.get_format_instructions(),
                    "paragraph": paragraph.strip()
                })

                response = model.invoke(final_prompt)

                # Parse the model response into a Pydantic object
                movie = parser.parse(response.content)

                # Save the result for Streamlit reruns
                st.session_state.movie_result = movie

            st.success("Movie information extracted successfully!")

        except Exception as e:
            st.error(
                "Could not extract movie information. "
                "Please check your API key and try again."
            )

            with st.expander("View error details"):
                st.code(str(e))


# -------------------- DISPLAY RESULTS --------------------

if st.session_state.get("movie_result") is not None:
    movie = st.session_state.movie_result

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">📊 Extracted Movie Information</div>',
        unsafe_allow_html=True
    )

    # Movie title
    st.title(f"🎞️ {movie.title}")

    # Main details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">📅 Release Year</div>
                <div class="info-value">
                    {movie.release_year if movie.release_year is not None else "Not mentioned"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">🎬 Director</div>
                <div class="info-value">
                    {movie.director or "Not mentioned"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">⭐ Rating</div>
                <div class="info-value">
                    {movie.rating if movie.rating is not None else "Not mentioned"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Genres
    st.markdown("### 🎭 Genres")

    if movie.genre:
        st.write("  ·  ".join(movie.genre))
    else:
        st.write("Not mentioned")

    # Cast
    st.markdown("### 👥 Cast")

    if movie.cast:
        for actor in movie.cast:
            st.markdown(f"- {actor}")
    else:
        st.write("Not mentioned")

    # Summary
    st.markdown("### 📖 Movie Summary")

    st.markdown(
        f"""
        <div class="info-card">
            {movie.summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Download result
    movie_text = f"""# {movie.title}

**Release Year:** {movie.release_year if movie.release_year is not None else "Not mentioned"}

**Director:** {movie.director or "Not mentioned"}

**Genres:** {", ".join(movie.genre) if movie.genre else "Not mentioned"}

**Cast:** {", ".join(movie.cast) if movie.cast else "Not mentioned"}

**Rating:** {movie.rating if movie.rating is not None else "Not mentioned"}

## Summary

{movie.summary}
"""

    st.download_button(
        label="📥 Download Movie Information",
        data=movie_text,
        file_name="movie_information.md",
        mime="text/markdown",
        use_container_width=True
    )

    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.pop("movie_result", None)
        st.rerun()


# -------------------- FOOTER --------------------

st.markdown("---")

st.caption(
    "CineExtract AI · Built with Streamlit, LangChain, Pydantic and Groq"
)
