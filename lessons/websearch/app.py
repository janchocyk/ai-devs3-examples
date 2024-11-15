from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time

# Import necessary modules
from enum import Enum
from dotenv import load_dotenv, find_dotenv

from openAIservice import OpenAIService
from websearch import WebSearchService

load_dotenv(find_dotenv())


class Role(str, Enum):
    """
    Enumeration of possible message roles.
    """
    user = 'user'
    assistant = 'assistant'
    system = 'system'

class Message(BaseModel):
    """
    Represents a chat message.

    Attributes:
        role (Role): The role of the message sender.
        content (str): The content of the message.
        name (Optional[str]): An optional name identifier.
    """
    role: Role
    content: str
    name: Optional[str] = None

class SearchResult(BaseModel):
    """
    Represents a search result.

    Attributes:
        url (str): The URL of the search result.
        title (str): The title of the search result.
        description (str): The description or snippet of the search result.
        content (Optional[str]): The scraped content from the URL.
    """
    url: str
    title: str
    description: str
    content: Optional[str] = None

class ChatRequest(BaseModel):
    """
    Represents a chat request containing a list of messages.

    Attributes:
        messages (List[Message]): The list of messages in the conversation.
    """
    messages: List[Message]

# Allowed domains
allowed_domains = [
    {'name': 'Wikipedia', 'url': 'en.wikipedia.org', 'scrappable': True},
    {'name': 'easycart', 'url': 'easycart.pl', 'scrappable': True},
    {'name': 'FS.blog', 'url': 'fs.blog', 'scrappable': True},
    {'name': 'arXiv', 'url': 'arxiv.org', 'scrappable': True},
    {'name': 'Instagram', 'url': 'instagram.com', 'scrappable': False},
    {'name': 'OpenAI', 'url': 'openai.com', 'scrappable': True},
    {'name': 'Brain overment', 'url': 'brain.overment.com', 'scrappable': True},
]

# Initialize FastAPI app
app = FastAPI()

# Initialize services
web_search_service = WebSearchService(allowed_domains)
openai_service = OpenAIService()

def answer_prompt(merged_results: List[Dict[str, Any]]) -> str:
    """
    Creates a system prompt incorporating the merged search results.

    Args:
        merged_results (List[Dict[str, Any]]): A list of search results with optional scraped content.

    Returns:
        str: The formatted system prompt including the search results.
    """
    prompt = "You are Alice, a helpful assistant. Use the following web search results to answer the user's question.\n\n"
    for result in merged_results:
        prompt += f"Title: {result.get('title', '')}\n"
        prompt += f"URL: {result.get('url', '')}\n"
        prompt += f"Description: {result.get('description', '')}\n"
        if result.get('content'):
            prompt += f"Content: {result['content']}\n"
        prompt += "\n"
    prompt += "Please provide a helpful answer based on the above information."
    return prompt

@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Handles incoming chat requests and generates a response.

    This endpoint processes the user's messages, determines if a web search is needed,
    performs the search and scraping if necessary, and generates a response using the OpenAI API.

    Args:
        chat_request (ChatRequest): The incoming chat request containing messages.

    Returns:
        dict: The response from the OpenAI API.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    print('Received request')

    messages = chat_request.messages

    try:
        # Find the latest user message
        latest_user_message = next((message for message in reversed(messages) if message.role == Role.user), None)
        if not latest_user_message:
            raise ValueError('No user message found')

        should_search = await web_search_service.is_web_search_needed(latest_user_message.content, openai_service)
        merged_results = []

        if should_search:
            queries, thoughts = await web_search_service.generate_queries(latest_user_message.content, openai_service)
            if queries:
                search_results = await web_search_service.search_web(queries)
                filtered_results = await web_search_service.score_results(search_results, latest_user_message.content, openai_service)
                urls_to_load = await web_search_service.select_resources_to_load(latest_user_message.content, filtered_results, openai_service)
                scraped_content = await web_search_service.scrape_urls(urls_to_load)
                # Merge the results
                for result in filtered_results:
                    scraped_item = next((item for item in scraped_content if item['url'] == result['url']), None)
                    if scraped_item:
                        result['content'] = scraped_item['content']
                    merged_results.append(result)

        prompt_with_results = answer_prompt(merged_results)
        all_messages = [{'role': 'system', 'content': prompt_with_results, 'name': 'Alice'}]
        all_messages.extend([message.dict() for message in messages])

        # Call the OpenAI API
        completion = openai_service.completion(all_messages, model="gpt-4o-mini")
        print(completion)
        return completion
    except Exception as e:
        print('Error in chat processing:', e)
        raise HTTPException(status_code=500, detail='An error occurred while processing your request')

@app.post("/api/chat-dummy")
async def chat_dummy(chat_request: ChatRequest):
    """
    Provides a dummy response for testing purposes.

    This endpoint mimics the structure of the OpenAI API response without making actual API calls.

    Args:
        chat_request (ChatRequest): The incoming chat request containing messages.

    Returns:
        dict: A dummy response mimicking the OpenAI API structure.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    messages = chat_request.messages

    try:
        latest_user_message = next((message for message in reversed(messages) if message.role == Role.user), None)
        if not latest_user_message:
            raise ValueError('No user message found')

        dummy_response = {
            'role': 'assistant',
            'content': f'This is a dummy response to: "{latest_user_message.content}"'
        }

        completion = {
            'id': 'dummy-completion-id',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'dummy-model',
            'choices': [
                {
                    'index': 0,
                    'message': dummy_response,
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }

        return completion
    except Exception as e:
        print('Error in chat processing:', e)
        raise HTTPException(status_code=500, detail='An error occurred while processing your request')
