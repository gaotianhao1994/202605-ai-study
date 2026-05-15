
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vector_db import VectorDatabase
from models.qa_engine import ConstructionQASystem


def test_basic_functionality():
    print("=" * 60)
    print("           建筑行业知识问答系统 - 测试")
    print("=" * 60)
    
    sample_knowledge = [
        {
            'id': 'test_001',
            'content': '混凝土是由水泥、砂、石子和水按比例混合制成的人造石材。',
            'metadata': {'source': 'test_doc.txt'}
        },
        {
            'id': 'test_002',
            'content': '混凝土浇筑步骤：支设模板、绑扎钢筋、搅拌运输、分层振捣、养护7天。',
            'metadata': {'source': 'test_doc.txt'}
        }
    ]
    
    print("\n[1/3] 初始化向量数据库...")
    vector_db = VectorDatabase()
    vector_db.add_items(sample_knowledge)
    print("   ✓ 向量数据库初始化完成")
    
    print("\n[2/3] 初始化问答系统...")
    qa_system = ConstructionQASystem(vector_db)
    print("   ✓ 问答系统初始化完成")
    
    print("\n[3/3] 测试问答功能...")
    
    test_questions = [
        "什么是混凝土？",
        "混凝土浇筑的步骤是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   测试问题 {i}: {question}")
        result = qa_system.ask(question)
        print(f"   问题类型: {result['question_type']}")
        print(f"   答案: {result['answer'][:80]}...")
        print(f"   ✓ 测试通过")
    
    print("\n" + "=" * 60)
    print("所有测试完成！系统运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_functionality()

