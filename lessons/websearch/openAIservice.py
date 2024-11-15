from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()

    def completion(
            self, 
            messages, 
            model: str = "gpt-4o-mini",
            json_mode: bool = False
        ):
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"} if json_mode else {"type": "text"}
            )

            return completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise error
