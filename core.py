from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-20b")

prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intelligent movie information extraction assistant.

Your task is to analyze the provided paragraph and extract useful movie information.

Extract the following details:
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
- Quick Summary (2–4 sentences explaining the movie's main premise and central conflict)

Instructions:
1. Extract information only from the provided paragraph.
2. If information is unavailable, write "Not mentioned".
3. Never invent facts, cast members, ratings, or plot details.
4. Use clear headings and bullet points.
5. Keep the output concise and avoid repetition.
6. Make the summary easy to understand, even for someone who has never watched the movie.

Return the result as readable text, not JSON."""
    ),
    (
        "human",
        "Extract the movie information from the following paragraph:\n\n{paragraph}"
    )
])

para=input("Give your paragraph: ")
 
final_prompt = prompt.invoke({
    "paragraph": para
})

response=model.invoke(final_prompt)

print(response.content)

