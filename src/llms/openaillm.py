from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

class OpenAILLM:
    def __init__(self, model, temperature=0):
        self.model = model
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def get_llm(self):
        return ChatOpenAI(model=self.model, temperature=0.0)