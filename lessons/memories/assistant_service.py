import uuid
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import asyncio

import prompts
from openai_sevice import OpenAIService
from memory_service import MemoryService

# Data classes for ParsingError and ShouldLearnResponse
class ParsingError:
    def __init__(self, error: str, result: bool):
        self.error = error
        self.result = result

class ShouldLearnResponse:
    def __init__(self, _thinking: str, add: Optional[List[str]] = None, update: Optional[List[Dict[str, str]]] = None):
        self._thinking = _thinking
        self.add = add or []
        self.update = update or []

class AssistantService:
    def __init__(self, openai_service: OpenAIService, memory_service: MemoryService):
        self.openai_service = openai_service
        self.memory_service = memory_service

    async def extract_queries(self, messages: List[Dict[str, Any]]) -> List[str]:
        try:
            thread = [
                {"role": "system", "content": prompts.extract_search_queries_prompt({"memoryStructure": prompts.memory_structure, "knowledge": prompts.default_knowledge})},
                *messages
            ]
            response = await self.openai_service.completion({"messages": thread, "jsonMode": True})
            result = self.openai_service.parse_json_response(response)

            if 'error' in result:
                raise Exception(result['error'])

            return result['q']

        except Exception as error:
            # Dodanie szczegółowej informacji o błędzie
            raise RuntimeError(f"Error in extract_queries: {error}") from error

    async def should_learn(self, messages: List[Dict[str, Any]], memories: str) -> Union[ShouldLearnResponse, ParsingError]:
        thread = [
            {
                "role": "system", 
                "content": prompts.should_learn_prompt(
                    memoryStructure=prompts.memory_structure,
                    knowledge=prompts.default_knowledge, 
                    memories=memories),
            },
            *messages
        ]
        try:
            thinking = await self.openai_service.completion({"messages": thread, "jsonMode": True})
            result = self.openai_service.parse_json_response(thinking)

            return ShouldLearnResponse(**result)
        except Exception as error:
            raise ValueError(error)

    async def learn(
            self, 
            messages: List[Dict[str, Any]], 
            should_learn_result: Union[ShouldLearnResponse, ParsingError], 
            memories: str
        ) -> str:
        if isinstance(should_learn_result, ParsingError) or (not should_learn_result.add and not should_learn_result.update):
            return '<memory_modifications>\n<no_changes>No memories were added, updated, or deleted.</no_changes>\n</memory_modifications>'

        try:
            add_results = await self.add_memories(should_learn_result.add, memories)
            update_results = await self.update_memories(should_learn_result.update, memories)
            memory_modifications = self.format_memory_modifications(add_results, update_results)

            return memory_modifications
        except Exception as error:
            error_message = str(error)
            raise ValueError(error_message)

    async def add_memories(self, memories_to_add: Optional[List[str]], memories: str) -> List[Dict[str, Any]]:
        if not memories_to_add:
            return []

        async def process_memory(memory_content):
            thread = [
                {
                    "role": "system", 
                    "content": prompts.learn_prompt(
                        memory_structure=prompts.memory_structure, 
                        knowledge=prompts.default_knowledge, 
                        memories=memories
                    )
                },
                {
                    "role": "user", 
                    "content": f"Please remember this: {memory_content}. Make sure to store it all and organize well in your knowledge structure."
                }
            ]
            try:
                thinking = await self.openai_service.completion({"messages": thread, "jsonMode": True})
                result = self.openai_service.parse_json_response(thinking)

                if 'error' in result:
                    
                    return {"status": "failed", "content": memory_content}

                memory = result
                memory['uuid'] = str(uuid.uuid4())
                memory['created_at'] = datetime.utcnow().isoformat()
                memory['updated_at'] = datetime.utcnow().isoformat()

                await self.memory_service.create_memory(memory)
                return {"status": "success", "name": memory.get('name'), "uuid": memory['uuid'], "content": memory_content}
            except Exception as error:
                error_message = str(error)
                return {"status": "failed", "content": memory_content}

        results = await asyncio.gather(*(process_memory(content) for content in memories_to_add))
        return results

    async def update_memories(self, memories_to_update: Optional[List[Dict[str, str]]], memories: str) -> List[Dict[str, Any]]:
        if not memories_to_update:
            return []

        async def process_update(update_memory):
            thread = [
                {
                    "role": "system", 
                    "content": prompts.update_memory_prompt(
                        memory_structure=prompts.memory_structure, 
                        knowledge=prompts.default_knowledge, 
                        memories=memories
                    )
                },
                {
                    "role": "user", 
                    "content": f"Please update this memory: {update_memory}"}
            ]
            try:
                thinking = await self.openai_service.completion({"messages": thread, "jsonMode": True})
                result = self.openai_service.parse_json_response(thinking)

                if 'error' in result:
                    return {"status": "failed", "content": str(update_memory)}

                if result.get('updating') and result.get('memory'):
                    await self.memory_service.update_memory(result['memory'])
                    return {"status": "success", "name": result['memory'].get('name'), "uuid": result['memory']['uuid'], "content": result['memory']['content']['text']}
                elif result.get('delete'):
                    for uuid_to_delete in result['delete']:
                        await self.memory_service.delete_memory(uuid_to_delete)
                    return {"status": "deleted", "uuids": result['delete']}
                else:
                    return {"status": "no_action", "content": str(update_memory)}
            except Exception as error:
                error_message = str(error)
                return {"status": "failed", "content": str(update_memory)}

        results = await asyncio.gather(*(process_update(update) for update in memories_to_update))
        return results

    def format_memory_modifications(self, add_results: List[Dict[str, Any]], update_results: List[Dict[str, Any]]) -> str:
        memory_modifications = "<memory_modifications>\n"

        for result in add_results:
            memory_modifications += f'<added status="{result["status"]}" name="{result.get("name", "")}" uuid="{result.get("uuid", "")}">{result["content"]}</added>\n'

        for result in update_results:
            if result["status"] == "success":
                memory_modifications += f'<updated status="{result["status"]}" name="{result["name"]}" uuid="{result["uuid"]}">{result["content"]}</updated>\n'
            elif result["status"] == "deleted":
                uuids = ','.join(result["uuids"]) if result.get("uuids") else ''
                memory_modifications += f'<deleted uuids="{uuids}" />\n'
            else:
                memory_modifications += f'<update_failed content="{result.get("content", "")}" />\n'

        memory_modifications += "</memory_modifications>"

        if memory_modifications == "<memory_modifications>\n</memory_modifications>":
            memory_modifications = "<memory_modifications>\n<no_changes>No memories were added, updated, or deleted.</no_changes>\n</memory_modifications>"

        return memory_modifications

    async def answer(self, config: Dict[str, Any], trace: LangfuseTraceClient):
        messages = config.get('messages', [])
        memories = config.get('memories', '')
        knowledge = config.get('knowledge', defaultKnowledge)
        learnings = config.get('learnings', '')
        rest_config = {k: v for k, v in config.items() if k not in ['messages', 'memories', 'knowledge', 'learnings']}

        system_message = {"role": "system", "content": f"As Alice, you're speaking to Adam. Answer based on the following memories:\n{memories} and general knowledge:\n{knowledge}. Learnings from the conversation:\n{learnings}"}
        messages_with_system = [system_message] + [msg for msg in messages if msg.get('role') != 'system']

        generation = self.langfuse_service.create_generation(trace, "Answer", {"messages": messages_with_system, **rest_config})

        try:
            completion = await self.openai_service.completion({
                **rest_config,
                "messages": messages_with_system
            })

            self.langfuse_service.finalize_generation(generation, completion.choices[0].message, completion.model, {
                "promptTokens": completion.usage.get("prompt_tokens") if completion.usage else None,
                "completionTokens": completion.usage.get("completion_tokens") if completion.usage else None,
                "totalTokens": completion.usage.get("total_tokens") if completion.usage else None,
            })
            return completion
        except Exception as error:
            self.langfuse_service.finalize_generation(generation, {"error": str(error)}, "unknown")
            raise

    async def get_relevant_context(self, query: str) -> str:
        similar_memories = await self.memory_service.search_similar_memories(query)
        return '\n\n'.join(memory['content']['text'] for memory in similar_memories)
