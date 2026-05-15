
import re
from typing import List, Dict, Tuple
import hashlib


class TextCleaner:
    @staticmethod
    def clean_text(text: str) -&gt; str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。；：""''（）【】、]', '', text)
        return text.strip()
    
    @staticmethod
    def split_sentences(text: str) -&gt; List[str]:
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_into_chunks(self, text: str, metadata: Dict = None) -&gt; List[Dict]:
        chunks = []
        words = text.split()
        start = 0
        
        while start &lt; len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            
            chunks.append({
                'id': chunk_id,
                'content': chunk_text,
                'metadata': metadata or {},
                'start_pos': start,
                'end_pos': end
            })
            
            start += self.chunk_size - self.overlap
        
        return chunks


class ConstructionKnowledgeExtractor:
    KEYWORDS = {
        'material': ['混凝土', '钢筋', '水泥', '砂浆', '砖块', '木材', '钢材', '玻璃'],
        'process': ['浇筑', '绑扎', '焊接', '抹灰', '防水', '保温', '吊装', '砌筑'],
        'standard': ['规范', '标准', '规程', '规定', '要求', '指标'],
        'safety': ['安全', '防护', '事故', '隐患', '应急', '救援']
    }
    
    @classmethod
    def extract_entities(cls, text: str) -&gt; Dict[str, List[str]]:
        entities = {}
        for category, keywords in cls.KEYWORDS.items():
            entities[category] = [kw for kw in keywords if kw in text]
        return entities
    
    @classmethod
    def classify_knowledge_type(cls, text: str) -&gt; str:
        scores = {}
        for category, keywords in cls.KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in text)
        
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return 'general'
        return max(scores.items(), key=lambda x: x[1])[0]


class KnowledgeBaseBuilder:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.chunker = Chunker()
        self.extractor = ConstructionKnowledgeExtractor()
    
    def process_document(self, doc_path: str, metadata: Dict = None) -&gt; List[Dict]:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned_text = self.cleaner.clean_text(text)
        chunks = self.chunker.split_into_chunks(cleaned_text, metadata)
        
        for chunk in chunks:
            chunk['entities'] = self.extractor.extract_entities(chunk['content'])
            chunk['knowledge_type'] = self.extractor.classify_knowledge_type(chunk['content'])
        
        return chunks
    
    def build_from_directory(self, dir_path: str) -&gt; List[Dict]:
        import os
        all_chunks = []
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt') or filename.endswith('.md'):
                file_path = os.path.join(dir_path, filename)
                chunks = self.process_document(file_path, {'source': filename})
                all_chunks.extend(chunks)
        
        return all_chunks

