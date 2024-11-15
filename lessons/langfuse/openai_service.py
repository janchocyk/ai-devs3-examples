from typing import List, Dict, Any

# from openai import OpenAI
from langfuse.openai import openai
from langfuse.decorators import observe
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# openai.flush_langfuse()
openai.langfuse_auth_check()

class OpenAIService:
    # def __init__(self):
        # Zakładamy, że klucz API jest już skonfigurowany przez zmienną środowiskową OPENAI_API_KEY
        # self.client = OpenAI()  # lub użyj zmiennej środowiskowej
    @staticmethod
    def completion(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wywołuje OpenAI API w celu uzyskania uzupełnienia rozmowy.
        
        Parametry:
        - config (Dict[str, Any]): Słownik zawierający parametry takie jak 'messages', 'model', 'stream', i 'jsonMode'.
        
        Zwraca:
        - Odpowiedź OpenAI w formacie JSON.
        """
        messages = config.get("messages", [])
        model = config.get("model", "gpt-4o-mini")
        stream = config.get("stream", False)
        json_mode = config.get("jsonMode", False)

        # Ustawienie odpowiedniego formatu odpowiedzi
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        
        # Wywołanie OpenAI API
        # response = self.client.chat.completions.create(
        response = openai.chat.completions.create(
            name="test",
            model=model,
            temperature=0,
            messages=messages,
            stream=stream,
            response_format=response_format
        )
        
        return response
