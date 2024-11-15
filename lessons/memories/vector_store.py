from uuid import uuid4
import asyncio

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import config


class QdrantService:
    def __init__(self):
        self.client = AsyncQdrantClient(path=config.QDRANT_STORAGE)
        
    async def create_collection(self, name: str, size: int):
        """
        Tworzy kolekcję wektorów i dodaje dokumenty.
        """
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )
        except Exception as e:
            raise ValueError(e)
        
    async def add_point(self, ids: str, vector: list[float], metadata: dict = {}):
        try:
            await self.client.upsert(
                collection_name="my_collection",
                points=[
                    PointStruct(
                        id=ids,
                        payload=metadata,
                        vector=vector,
                    )
                    for i in range(100)
                ],
            )
            return "Point added."
        except Exception as e:
            raise ValueError(e)

    async def similarity_search(self, name: str, query_vector, k: int) -> dict:
        points =self.client.query_points(
            collection_name=name,
            query=query_vector,
            with_vectors=True,
            with_payload=True,
            limit=k
        ).points
        return points
