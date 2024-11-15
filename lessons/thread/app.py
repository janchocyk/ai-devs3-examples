from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import asyncio

from lessons.thread.open_ai_service import OpenAIService

# Initialize OpenAI API key
openai.api_key = 'your-api-key'  # Replace with your actual API key

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAIService
openai_service = OpenAIService()

# Initialize previous_summarization
previous_summarization = ""

# Define Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: Message

# Function to generate summarization based on the current turn and previous summarization
async def generate_summarization(user_message: Message, assistant_response: Message) -> str:
    global previous_summarization
    summarization_prompt = Message(
        role="system",
        content=f"""Please summarize the following conversation in a concise manner, incorporating the previous summary if available:
        <previous_summary>{previous_summarization or "No previous summary"}</previous_summary>
        <current_turn> Adam: {user_message.content}\nAlice: {assistant_response.content} </current_turn>
        """
    )

    response = await openai_service.completion(
        [summarization_prompt.dict(), {"role": "user", "content": "Please create/update our conversation summary."}],
        model="gpt-4",
        stream=False
    )

    content = response.choices[0].message.get('content', "No conversation history")
    return content

# Function to create system prompt
def create_system_prompt(summarization: str) -> Message:
    content = f"""You are Alice, a helpful assistant who speaks using as few words as possible.

{'Here is a summary of the conversation so far:\n<conversation_summary>\n  ' + summarization + '\n</conversation_summary>' if summarization else ''}

Let's chat!"""
    return Message(
        role="system",
        content=content
    )

# Chat endpoint POST /api/chat
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    global previous_summarization
    message = chat_request.message

    try:
        system_prompt = create_system_prompt(previous_summarization)

        assistant_response = await openai_service.completion(
            [system_prompt.dict(), message.dict()],
            model="gpt-4",
            stream=False
        )

        # Generate new summarization
        previous_summarization = await generate_summarization(message, assistant_response.choices[0].message)
        
        return assistant_response
    except Exception as e:
        print('Error in OpenAI completion:', e)
        raise HTTPException(status_code=500, detail='An error occurred while processing your request')

# Demo endpoint POST /api/demo
@app.post("/api/demo")
async def demo_endpoint():
    global previous_summarization
    demo_messages = [
        Message(content="Hi! I'm Adam", role="user"),
        Message(content="How are you?", role="user"),
        Message(content="Do you know my name?", role="user")
    ]

    assistant_response = None

    for message in demo_messages:
        print('--- NEXT TURN ---')
        print('Adam:', message.content)

        try:
            system_prompt = create_system_prompt(previous_summarization)

            assistant_response = await openai_service.completion(
                [system_prompt.dict(), message.dict()],
                model="gpt-4",
                stream=False
            )

            print('Alice:', assistant_response.choices[0].message.get('content', ''))

            # Generate new summarization
            previous_summarization = await generate_summarization(message, assistant_response.choices[0].message)
        except Exception as e:
            print('Error in OpenAI completion:', e)
            raise HTTPException(status_code=500, detail='An error occurred while processing your request')

    return assistant_response
