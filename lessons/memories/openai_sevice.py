import logging
import json
from typing import List, Dict, Union, Any, AsyncGenerator, Optional

from openai import OpenAI
import tiktoken

class OpenAIService:
    def __init__(self):
        self.openai = OpenAI()
        self.tokenizers = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"

    def get_tokenizer(self, model_name: str):
        if model_name not in self.tokenizers:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logging.warning(f"Model {model_name} not found. Using default encoding 'cl100k_base'.")
                encoding = tiktoken.get_encoding("cl100k_base")
            self.tokenizers[model_name] = encoding
        return self.tokenizers[model_name]

    def count_tokens(self, messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
        encoding = self.get_tokenizer(model)
        # Adjust model name for token counting
        if model == "gpt-3.5-turbo":
            model = "gpt-3.5-turbo-0301"
        elif model == "gpt-4":
            model = "gpt-4-0314"

        # Set tokens per message based on the model
        if model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in ["gpt-4-0314", "gpt-4"]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"Token counting not implemented for model {model}.")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <im_start>assistant
        return num_tokens

    def completion(self, config: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        messages = config.get('messages', [])
        model = config.get('model', 'gpt-4o-mini')
        stream = config.get('stream', False)
        json_mode = config.get('jsonMode', False)
        max_tokens = config.get('maxTokens', 4096)

        try:
            response = self.openai.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                response_format = {"type": "json_object"} if json_mode else {"type": "text"},
                temperature=0,
            )
            return response
        except Exception as e:
            logging.error("Error in OpenAI completion:", exc_info=True)
            raise e

    def is_stream_response(self, response: Any) -> bool:
        return hasattr(response, '__iter__') and not hasattr(response, '__len__')

    def parse_json_response(self, response: Dict[str, Any]) -> Union[Dict[str, Any], Dict[str, Any]]:
        try:
            content = response.choices[0].message.content
            parsed_content = json.loads(content)
            return parsed_content
        except Exception as e:
            logging.error('Error parsing JSON response:', exc_info=True)
            return {'error': 'Failed to process response', 'result': False}

    def create_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error("Error creating embedding:", exc_info=True)
            raise ValueError(e)
