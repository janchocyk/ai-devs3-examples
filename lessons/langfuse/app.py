from typing import List, Optional
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat_service import ChatService


app = FastAPI()
chat_service = ChatService()

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None


@app.post("/api/chat/")
async def chat_endpoint(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())

    try:
        # Prepare initial messages
        all_messages = [
            {"role": "system", "content": "You are a helpful assistant.", "name": "Alice"},
            *request.messages
        ]
        
        generated_messages = []

        # Main Completion - Answer user's question
        # main_span = langfuse_service.create_span(trace, "Main Completion", all_messages)
        main_completion = chat_service.completion(all_messages, "gpt-4o-mini")
        # langfuse_service.finalize_span(main_span, "Main Completion", all_messages, main_completion)
        main_message = main_completion.choices[0].message
        all_messages.append(main_message)
        generated_messages.append(main_message)

        return main_completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
