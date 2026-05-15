
from typing import List, Dict, Any
import re


class QuestionAnalyzer:
    QUESTION_TYPES = {
        'definition': ['什么是', '什么叫', '定义', '概念', '解释'],
        'how_to': ['如何', '怎样', '怎么', '方法', '步骤', '流程'],
        'why': ['为什么', '原因', '为何', '导致'],
        'material': ['材料', '用什么', '选用', '材料选择'],
        'safety': ['安全', '注意事项', '防护', '预防'],
        'standard': ['规范', '标准', '规定', '要求']
    }
    
    @classmethod
    def classify_question(cls, question: str) -&gt; str:
        for q_type, keywords in cls.QUESTION_TYPES.items():
            if any(kw in question for kw in keywords):
                return q_type
        return 'general'
    
    @classmethod
    def extract_keywords(cls, question: str) -&gt; List[str]:
        keywords = []
        for q_type, type_keywords in cls.QUESTION_TYPES.items():
            keywords.extend([kw for kw in type_keywords if kw in question])
        return list(set(keywords))


class AnswerGenerator:
    TEMPLATES = {
        'definition': '根据知识库，{0}',
        'how_to': '{0}的步骤如下：{1}',
        'why': '{0}的原因是：{1}',
        'material': '常用的材料包括：{0}',
        'safety': '安全注意事项：{0}',
        'standard': '根据相关规范要求：{0}',
        'general': '关于您的问题，相关信息如下：{0}'
    }
    
    @classmethod
    def generate(cls, question: str, question_type: str, contexts: List[Dict]) -&gt; Dict[str, Any]:
        context_texts = [ctx['content'] for ctx in contexts]
        combined_context = '\n\n'.join(context_texts[:3])
        
        template = cls.TEMPLATES.get(question_type, cls.TEMPLATES['general'])
        
        if question_type == 'definition':
            answer = template.format(combined_context)
        elif question_type == 'how_to':
            answer = template.format(question, combined_context)
        else:
            answer = template.format(combined_context)
        
        return {
            'question': question,
            'question_type': question_type,
            'answer': answer,
            'sources': [ctx['metadata'].get('source', 'unknown') for ctx in contexts],
            'confidence': sum(ctx.get('score', 0) for ctx in contexts) / len(contexts) if contexts else 0
        }


class ConstructionQASystem:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.analyzer = QuestionAnalyzer()
        self.generator = AnswerGenerator()
    
    def ask(self, question: str, top_k: int = 5) -&gt; Dict[str, Any]:
        question_type = self.analyzer.classify_question(question)
        search_results = self.vector_db.search(question, top_k=top_k)
        
        contexts = []
        for item_id, score, data in search_results:
            contexts.append({
                'content': data['content'],
                'metadata': data['metadata'],
                'score': float(score)
            })
        
        return self.generator.generate(question, question_type, contexts)

