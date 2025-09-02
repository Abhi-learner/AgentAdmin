from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Any, Dict
import os

class GroqLLM:
    def __init__(self, model: str, **default_params: Any) :
        """
        Example:
          GroqLLM("llama-3.1-70b-versatile", temperature=0, timeout=60)
        You can also set response_format here if you want JSON mode by default:
          GroqLLM("llama-3.1-70b-versatile", response_format={"type":"json_object"})
        """
        # Load env once at construction
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key  # ensure SDK sees it

        self.model = model
        self.default_params: Dict[str, Any] = {"model": model}
        self.default_params.update(default_params)

    def get_llm(self, **overrides: Any) -> ChatGroq:
        """
        Create a ChatGroq with merged params.
        Call-site can override anything, e.g.:
          get_llm(response_format={"type":"json_object"}, temperature=0)
        """
        params = {**self.default_params, **overrides}
        return ChatGroq(**params)

    # Optional convenience helpers
    def json_mode(self, **overrides: Any) -> ChatGroq:
        """
        Force JSON-only responses (no tool/function-calling).
        """
        base = {"response_format": {"type": "json_object"}}
        base.update(overrides)
        return self.get_llm(**base)

    def text_mode(self, **overrides: Any) -> ChatGroq:
        """
        Plain text responses.
        """
        return self.get_llm(**overrides)
