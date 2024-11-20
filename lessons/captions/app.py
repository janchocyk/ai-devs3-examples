from dataclasses import dataclass
from typing import List, Dict, Optional
import openai
import requests
import base64
import re
import json
import os
from pathlib import Path

from prompts import extract_image_context_system_message, refine_description_system_message, preview_image_system_message
from openai_service import OpenAIService

# Define the Image dataclass
@dataclass
class Image:
    alt: str
    url: str
    context: str
    description: str
    preview: str
    base64: str
    name: str

def extract_images(article: str) -> List[Image]:
    image_regex = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    matches = image_regex.findall(article)

    images = []

    for alt, url in matches:
        try:
            name = url.split('/')[-1]
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: {response.status_code} {response.reason}")
                continue
            array_buffer = response.content  # This is bytes
            base64_data = base64.b64encode(array_buffer).decode('utf-8')

            images.append(
                Image(
                    alt=alt,
                    url=url,
                    context='',
                    description='',
                    preview='',
                    base64=base64_data,
                    name=name,
                )
            )
        except Exception as e:
            print(f"Error processing image {url}: {e}")
    return images

def preview_image(image: Image) -> Dict[str, str]:
    # Convert the base64 data back to bytes
    # image_bytes = base64.b64decode(image.base64)

    user_message_content = f"Describe the image {image.name} concisely. Focus on the main elements and overall composition. Return the result in JSON format with only 'name' and 'preview' properties."
    messages = [
        {
            "role": "system", 
            "content": preview_image_system_message.get('content')
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": user_message_content
                },
                {
                    "type": "image_url",
                    "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{image.base64}",
                        },
                },
            ]
        },
    ]
    model_config = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "json_mode": True,
        "name": "captions: preview_image"
    }
    response = OpenAIService.completion(config=model_config)
    response_text = response.choices[0].message.content
    print("captions: preview_image")
    print(response_text)
    try:
        result = json.loads(response_text)
    except Exception:
        pattern = r"```json\s*(\{.*?\})\s*```"
        # Wyszukiwanie dopasowania
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            # Wyciągnięcie zawartości JSON
            final_string = match.group(1)
            
            # Wyświetlenie final_string (opcjonalne)
            print("Final JSON String:")
            print(final_string)
            
            # Parsowanie JSON do obiektu Python
            try:
                result = json.loads(final_string)
                print("\nParsed JSON Object:")
                print(result)
            except json.JSONDecodeError as e:
                print("Błąd podczas parsowania JSON:", e)
                result = {'name': "error", 'previev': e}
        else:
            print("Nie znaleziono bloku JSON w oryginalnym stringu.")
    
    
    return {'name': result.get('name', image.name), 'preview': result.get('preview', '')}

def get_image_context(title: str, article: str, images: List[Image]) -> Dict[str, List[Dict[str, str]]]:
    user_message_content = f"Title: {title}\n\n{article}"
    system_message = extract_image_context_system_message(images)
    messages = [
        {"role": "system", "content": system_message.get('content')},
        {"role": "user", "content": user_message_content},
    ]
    model_config = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "json_mode": True,
        "name": "captions: extract_image_context"
    }
    response = OpenAIService.completion(config=model_config)
    response_text = response.choices[0].message.content
    print("captions: extract_image_context")
    print(response_text)
    result = json.loads(response_text)

    # Generate previews for all images
    previews = [preview_image(image) for image in images]

    # Merge context and preview information
    merged_results = []
    for context_image in result.get('images', []):
        preview = next((p for p in previews if p['name'] == context_image['name']), None)
        merged_results.append({
            'name': context_image['name'],
            'context': context_image['context'],
            'preview': preview['preview'] if preview else ''
        })

    return {'images': merged_results}

def refine_description(image: Image) -> Image:
    # Convert the base64 data back to bytes
    image_bytes = base64.b64decode(image.base64)

    user_message_content = f"""
        Write a description of the image {image.name}. I have some <context>{image.context}</context>
        that should be useful for understanding the image in a better way. An initial preview of the image is:
        <preview>{image.preview}</preview>. A good description briefly describes what is on the image, and uses
        the context to make it more relevant to the article. The purpose of this description is for summarizing the
        article, so we need just an essence of the image considering the context, not a detailed description of what is on the image."""

    messages = [
        {"role": "system", "content": refine_description_system_message.get('content')},
        {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{image.base64}",
                        },
                },
                {
                    "type": "text", 
                    "text": user_message_content
                },
            ]
        },
    ]
    model_config = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "json_mode": False,
        "name": "captions: refine_description"
    }
    response = OpenAIService.completion(config=model_config)
    response_text = response.choices[0].message.content
    print("captions: refine_description")
    print(response_text)
    image.description = response_text.strip()
    return image

def process_and_summarize_images(title: str, path: str):
    # Read the article file
    with open(path, 'r', encoding='utf-8') as f:
        article = f.read()

    # Extract images from the article
    images = extract_images(article)
    print('Number of images found:', len(images))

    contexts = get_image_context(title, article, images)
    print('Number of image metadata found:', len(contexts['images']))

    # Process each image
    processed_images = []
    for image in images:
        context_data = next((ctx for ctx in contexts['images'] if ctx['name'] == image.name), {})
        image.context = context_data.get('context', '')
        image.preview = context_data.get('preview', '')
        processed_image = refine_description(image)
        processed_images.append(processed_image)

    # Prepare and save the summarized images (excluding base64 data)
    described_images = []
    for image in processed_images:
        image_data = image.__dict__.copy()
        del image_data['base64']
        described_images.append(image_data)

    with open('descriptions.json', 'w', encoding='utf-8') as f:
        json.dump(described_images, f, ensure_ascii=False, indent=2)

    # Prepare and save the final data (only url and description)
    captions = [{'url': image.url, 'description': image.description} for image in processed_images]
    with open('captions.json', 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    # Log completion messages
    print('Final data saved to captions.json')

if __name__ == '__main__':
    # Adjust the title and article path as needed
    process_and_summarize_images(
        'Lesson #0201 — Audio i interfejs głosowy',
        os.path.join(os.path.dirname(__file__), 'article.md')
    )
