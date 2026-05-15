
import numpy as np
from typing import List, Dict, Tuple
import json
import os


class SimpleEmbeddingModel:
    def __init__(self, dim: int = 128):
        self.dim = dim
        np.random.seed(42)
    
    def encode(self, text: str) -&gt; np.ndarray:
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm &gt; 0 else embedding


class VectorDatabase:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.contents: Dict[str, str] = {}
    
    def add_item(self, item_id: str, content: str, metadata: Dict = None):
        vector = self.embedding_model.encode(content)
        self.vectors[item_id] = vector
        self.contents[item_id] = content
        self.metadata[item_id] = metadata or {}
    
    def add_items(self, items: List[Dict]):
        for item in items:
            self.add_item(
                item_id=item['id'],
                content=item['content'],
                metadata=item.get('metadata', {})
            )
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -&gt; float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def search(self, query: str, top_k: int = 5) -&gt; List[Tuple[str, float, Dict]]:
        query_vector = self.embedding_model.encode(query)
        scores = []
        
        for item_id, vector in self.vectors.items():
            score = self.cosine_similarity(query_vector, vector)
            scores.append((item_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        
        return [
            (item_id, score, {
                'content': self.contents[item_id],
                'metadata': self.metadata[item_id]
            })
            for item_id, score in top_results
        ]
    
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        vector_data = {k: v.tolist() for k, v in self.vectors.items()}
        with open(os.path.join(save_dir, 'vectors.json'), 'w', encoding='utf-8') as f:
            json.dump(vector_data, f)
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)
        
        with open(os.path.join(save_dir, 'contents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.contents, f)
    
    def load(self, load_dir: str):
        with open(os.path.join(load_dir, 'vectors.json'), 'r', encoding='utf-8') as f:
            vector_data = json.load(f)
        self.vectors = {k: np.array(v, dtype=np.float32) for k, v in vector_data.items()}
        
        with open(os.path.join(load_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        with open(os.path.join(load_dir, 'contents.json'), 'r', encoding='utf-8') as f:
            self.contents = json.load(f)

