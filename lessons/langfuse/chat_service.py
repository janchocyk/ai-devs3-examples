from openai_service import OpenAIService
from typing import List, Dict, Any

class ChatService:

    @staticmethod
    def completion(messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """
        Wywołuje metodę completion z OpenAIService, przekazując listę wiadomości i model.
        """
        return OpenAIService.completion({
            "messages": messages,
            "model": model,
            "stream": False,
            "jsonMode": False
        })
