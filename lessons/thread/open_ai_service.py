from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()

    def completion(self, messages, model: str = "gpt-4o"):
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise error