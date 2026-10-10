import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def get_llm(temperature: float = 0.7, streaming: bool = False) -> ChatGroq:
    return ChatGroq(model=MODEL, temperature=temperature, streaming=streaming)