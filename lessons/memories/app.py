from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import logging

from openai_sevice import OpenAIService
from assistant_service import AssistantService
from memory_service import MemoryService

# Pydantic models for request and response
class ChatCompletionMessageParam(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    conversation_id: str = None

# Initialize services
openaiService = OpenAIService()
memoryService = MemoryService('memory/memories', openaiService)
assistantService = AssistantService(openaiService, memoryService)

# Initialize FastAPI app
app = FastAPI()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    messages = request.messages
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Filter out 'system' role messages
    messages = [msg for msg in messages if msg.role != 'system']

    try:
        queries = await assistantService.extract_queries(messages)
        memories = await memoryService.recall(queries)
        should_learn = await assistantService.should_learn(messages, memories)
        learnings = await assistantService.learn(messages, should_learn, memories)
        answer = await assistantService.answer(
            {
                'messages': messages,
                'memories': memories,
                'knowledge': knowledge,
                'learnings': learnings
            }
        )

        return {**answer, 'conversation_id': conversation_id}

    except Exception as error:
        logging.error('Error in chat processing:', exc_info=True)
        raise HTTPException(status_code=500, detail='An error occurred while processing your request')

# @app.post("/api/sync")
# async def sync_endpoint():
#     trace = langfuseService.createTrace({
#         'id': str(uuid.uuid4()),
#         'name': 'Sync Memories',
#         'sessionId': str(uuid.uuid4())
#     })

#     try:
#         changes = await memoryService.sync_memories(trace)
#         await langfuseService.finalizeTrace(trace, {}, changes)
#         await langfuseService.flushAsync()
#         return changes
#     except Exception as error:
#         await langfuseService.finalizeTrace(trace, {}, {'error': 'An error occurred while syncing memories'})
#         logging.error('Error in memory synchronization:', exc_info=True)
#         raise HTTPException(status_code=500, detail='An error occurred while syncing memories')

# @app.on_event("shutdown")
# async def shutdown_event():
#     await langfuseService.shutdownAsync()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
