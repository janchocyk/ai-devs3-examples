import os
import json
import uuid
import yaml
import asyncio
import logging
from slugify import slugify
from typing import List, Dict, Any, Optional
from datetime import datetime
import subprocess

from openai_sevice import OpenAIService
from vector_store import QdrantService
from config import config
 

class Memory:
    def __init__(self, uuid: str, category: str, subcategory: str, name: str,
                 content: Dict[str, Any], metadata: Dict[str, Any],
                 created_at: str, updated_at: str):
        self.uuid = uuid
        self.category = category
        self.subcategory = subcategory
        self.name = name
        self.content = content
        self.metadata = metadata
        self.created_at = created_at
        self.updated_at = updated_at

class MemoryService:
    def __init__(self, openai_service: OpenAIService = None):
        self.base_dir = config.MEMORIES
        self.index_file = os.path.join(self.base_dir, 'index.jsonl')
        self.openai_service = openai_service
        self.vector_store = QdrantService()  # Adjusted dimension for text-embedding-3-large
        if not self.vector_store.client.collection_exist(collection_name='memory'):
            self.vector_store.create_collection(name='memory', size=3072)
        self.ensure_directories()

    def ensure_directory_exists(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)

    def ensure_directories(self):
        categories = ['profiles', 'preferences', 'resources', 'events', 'locations', 'environment']
        subcategories = {
            'profiles': ['basic', 'work', 'development', 'relationships'],
            'preferences': ['hobbies', 'interests'],
            'resources': ['books', 'movies', 'music', 'videos', 'images', 'apps', 'devices', 'courses', 'articles', 'communities', 'channels', 'documents', 'notepad'],
            'events': ['personal', 'professional'],
            'locations': ['places', 'favorites'],
            'environment': ['current']
        }

        for category in categories:
            self.ensure_directory_exists(os.path.join(self.base_dir, category))
            for subcategory in subcategories.get(category, []):
                self.ensure_directory_exists(os.path.join(self.base_dir, category, subcategory))

    def append_to_index(self, memory: Memory):
        index_entry = json.dumps(memory.__dict__) + '\n'
        with open(self.index_file, 'a', encoding='utf-8') as f:
            f.write(index_entry)

    def json_to_markdown(self, memory: Memory) -> str:
        content = memory.content
        frontmatter_data = memory.__dict__.copy()
        del frontmatter_data['content']

        yaml_frontmatter = yaml.dump(frontmatter_data, allow_unicode=True)

        markdown_content = f"---\n{yaml_frontmatter}---\n\n{content['text']}"

        # Add hashtags at the end of the file
        tags = frontmatter_data.get('metadata', {}).get('tags', [])
        if tags:
            markdown_content += '\n\n' + ' '.join(f"#{tag.replace(' ', '_')}" for tag in tags)

        return markdown_content

    def markdown_to_json(self, markdown: str) -> Memory:
        parts = markdown.split('---')
        if len(parts) < 3:
            raise ValueError("Invalid markdown format")
        _, frontmatter, content = parts
        data = yaml.safe_load(frontmatter.strip())

        # Split content into main text and hashtags
        content_parts = content.strip().split('\n\n')
        main_content = content_parts[0]
        hashtags = ' '.join(content_parts[1:]).strip()

        # Ensure tags in metadata match those at the end of the file
        if hashtags:
            tags_from_content = [tag.replace('#', '').replace('_', ' ') for tag in hashtags.split(' ')]
            data.setdefault('metadata', {})
            data['metadata']['tags'] = list(set(data['metadata'].get('tags', []) + tags_from_content))

        data['content'] = {
            'text': main_content,
            'hashtags': hashtags  # Store hashtags separately if needed
        }

        return Memory(**data)

    def get_memory_file_path(self, memory: Memory) -> str:
        slugified_name = slugify(memory.name, lowercase=True)
        return os.path.join(
            self.base_dir,
            slugify(memory.category, lowercase=True),
            slugify(memory.subcategory, lowercase=True),
            f"{slugified_name}.md"
        )

    async def create_memory(self, memory_data: Dict[str, Any]) -> Memory:
        new_memory = Memory(
            uuid=str(uuid.uuid4()),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            **memory_data
        )
        try:
            embedding = await self.openai_service.create_embedding(new_memory.content['text'])

            # Add the embedding to the vector store
            self.vector_store.add_point(new_memory.uuid, embedding)

            dir_path = os.path.join(
                self.base_dir,
                slugify(new_memory.category, lowercase=True),
                slugify(new_memory.subcategory, lowercase=True)
            )
            self.ensure_directory_exists(dir_path)

            file_path = self.get_memory_file_path(new_memory)
            markdown_content = self.json_to_markdown(new_memory)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self.append_to_index(new_memory)

            return new_memory
        except Exception as e:
            raise ValueError(e)

    async def get_memory(self, uuid_str: str) -> Optional[Memory]:
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                memories = [json.loads(line) for line in f if line.strip()]
            memory_data = next((m for m in memories if m['uuid'] == uuid_str), None)
            if not memory_data:
                return None

            file_path = self.get_memory_file_path(Memory(**memory_data))
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            return self.markdown_to_json(file_content)
        except Exception as e:
            logging.error(f"Error reading memory: {e}")
            return None

    async def update_memory(self, memory: Memory) -> Memory:
        try:
            memory.updated_at = datetime.utcnow().isoformat()
            file_path = self.get_memory_file_path(memory)
            markdown_content = self.json_to_markdown(memory)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Update the embedding if the content has changed
            old_memory = await self.get_memory(memory.uuid)
            if old_memory and old_memory.content['text'] != memory.content['text']:
                new_embedding = await self.openai_service.create_embedding(memory.content['text'])
                self.vector_store.update(new_embedding, memory.uuid)

            # Update index file
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_lines = f.readlines()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                for line in index_lines:
                    if line.strip():
                        indexed_memory = json.loads(line)
                        if indexed_memory['uuid'] == memory.uuid:
                            f.write(json.dumps(memory.__dict__) + '\n')
                        else:
                            f.write(line)
            return memory
        except Exception as e:
            logging.error(f"Error updating memory: {e}")
            raise e

    async def search_memories(self, query: str) -> List[Memory]:
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                memories = [json.loads(line) for line in f if line.strip()]
            return [
                Memory(**memory) for memory in memories
                if query.lower() in memory['name'].lower() or query.lower() in memory['content']['text'].lower()
            ]
        except Exception as e:
            logging.error(f"Error searching memories: {e}")
            return []

    async def search_similar_memories(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self.openai_service.create_embedding(query)
            similar_results = self.vector_store.search(query_embedding, k)
            if not similar_results:
                logging.info('No similar memories found.')
                return []
            memories = await asyncio.gather(*(self.get_memory(result['id']) for result in similar_results))
            return [
                {**memory.__dict__, 'similarity': similar_results[index]['similarity']}
                for index, memory in enumerate(memories) if memory
            ]
        except Exception as e:
            logging.error(f"Error searching similar memories: {e}")
            return []

    async def delete_memory(self, uuid_str: str) -> bool:
        try:
            memory = await self.get_memory(uuid_str)
            if not memory:
                return False

            file_path = self.get_memory_file_path(memory)
            os.remove(file_path)

            # Update the index file
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_lines = f.readlines()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                for line in index_lines:
                    if line.strip():
                        indexed_memory = json.loads(line)
                        if indexed_memory['uuid'] != uuid_str:
                            f.write(line)
            return True
        except Exception as e:
            logging.error(f"Error deleting memory: {e}")
            return False

    async def recall(self, queries: List[str]) -> str:
        try:
            recalled_memories_lists = await asyncio.gather(*(self.search_similar_memories(query) for query in queries))
            unique_memories = {memory['uuid']: memory for memories in recalled_memories_lists for memory in memories}.values()

            if not unique_memories:
                result = '<recalled_memories>No relevant memories found.</recalled_memories>'
            else:
                formatted_memories = '\n'.join(self.format_memory(Memory(**memory)) for memory in unique_memories)
                result = f"<recalled_memories>\n{formatted_memories}</recalled_memories>"

            logging.info(f'Recalled memories: {result}')
            return result
        except Exception as e:
            error_message = str(e)
            error_result = f"<recalled_memories>Error: {error_message}</recalled_memories>"
            raise ValueError(error_result)

    def format_memory(self, memory: Memory) -> str:
        urls = memory.metadata.get('urls', [])
        urls_str = f"\nURLs: {', '.join(urls)}" if urls else ''
        return (
            f'<memory uuid="{memory.uuid}" name="{memory.name}" category="{memory.category}" '
            f'subcategory="{memory.subcategory}" lastmodified="{memory.updated_at}">'
            f'{memory.content["text"]}{urls_str}</memory>'
        )

    async def sync_memories(self, trace: LangfuseTraceClient) -> Dict[str, List[str]]:
        git_diff = self.get_git_diff()
        changes = self.parse_git_diff(git_diff)

        logging.info(f"Changes detected: {changes}")
        added = []
        modified = []
        deleted = []

        for file in changes['added']:
            memory = await self.add_memory_from_file(file)
            if memory:
                added.append(memory.uuid)

        for file in changes['modified']:
            logging.info(f'Updating file {file}')
            memory = await self.update_memory_from_file(file)
            if memory:
                modified.append(memory.uuid)

        for file in changes['deleted']:
            success = await self.delete_memory_by_file(file)
            if success:
                deleted.append(file)

        self.langfuse_service.create_event(trace, "SyncMemories", {"added": added, "modified": modified, "deleted": deleted})
        return {"added": added, "modified": modified, "deleted": deleted}

    def get_git_diff(self) -> str:
        command = ['git', 'diff', '--name-status', 'HEAD']
        result = subprocess.run(command, cwd=self.base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Git diff error: {result.stderr}")
            return ''
        return result.stdout

    def parse_git_diff(self, diff: str) -> Dict[str, List[str]]:
        changes = {'added': [], 'modified': [], 'deleted': []}
        lines = diff.strip().split('\n')
        for line in lines:
            if not line:
                continue
            status, file = line.split('\t', 1)
            if not file.endswith('.md'):
                continue
            if status == 'A':
                changes['added'].append(file)
            elif status == 'M':
                changes['modified'].append(file)
            elif status == 'D':
                changes['deleted'].append(file)
        return changes

    async def add_memory_from_file(self, file: str) -> Optional[Memory]:
        try:
            file_path = os.path.join(self.base_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            memory = self.markdown_to_json(content)
            return await self.create_memory(memory.__dict__, LangfuseTraceClient())
        except Exception as e:
            logging.error(f"Error adding memory from file {file}: {e}")
            return None

    async def update_memory_from_file(self, file: str) -> Optional[Memory]:
        try:
            file_path = os.path.join(self.base_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            updated_memory = self.markdown_to_json(content)
            return await self.update_memory(updated_memory, LangfuseTraceClient())
        except Exception as e:
            logging.error(f"Error updating memory from file {file}: {e}")
            return None

    async def delete_memory_by_file(self, file: str) -> bool:
        try:
            file_path = os.path.join(self.base_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            memory = self.markdown_to_json(content)
            return await self.delete_memory(memory.uuid)
        except Exception as e:
            logging.error(f"Error deleting memory by file {file}: {e}")
            return False
