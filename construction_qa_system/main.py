
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_preprocessor import KnowledgeBaseBuilder
from models.vector_db import VectorDatabase
from models.qa_engine import ConstructionQASystem


def create_sample_knowledge():
    sample_docs = [
        {
            'id': 'conc_001',
            'content': '混凝土是由水泥、砂、石子和水按一定比例混合，经搅拌、浇筑、养护而成的人造石材。其强度等级通常用C表示，如C30表示抗压强度为30MPa。',
            'metadata': {'source': '混凝土基础知识.txt', 'category': 'material'}
        },
        {
            'id': 'conc_002',
            'content': '混凝土浇筑的基本步骤：1.支设模板；2.钢筋绑扎；3.混凝土搅拌运输；4.分层浇筑振捣；5.养护至少7天；6.拆模。',
            'metadata': {'source': '混凝土施工规范.txt', 'category': 'process'}
        },
        {
            'id': 'conc_003',
            'content': '防止混凝土开裂的措施：1.控制水灰比；2.选用合适的水泥品种；3.加强养护保湿；4.设置伸缩缝；5.控制浇筑温度。',
            'metadata': {'source': '混凝土防裂技术.txt', 'category': 'process'}
        },
        {
            'id': 'rebar_001',
            'content': '钢筋按强度等级分为HPB300、HRB400、HRB500等。绑扎搭接长度应满足规范要求，HRB400钢筋的搭接长度一般为35-40倍直径。',
            'metadata': {'source': '钢筋工程手册.txt', 'category': 'material'}
        },
        {
            'id': 'safety_001',
            'content': '施工现场安全注意事项：1.佩戴安全帽；2.高空作业系安全带；3.临时用电接地保护；4.消防器材配置齐全；5.定期安全检查。',
            'metadata': {'source': '安全施工规范.txt', 'category': 'safety'}
        }
    ]
    return sample_docs


def main():
    print("=" * 60)
    print("           建筑行业知识问答系统")
    print("=" * 60)
    
    print("\n[1/4] 初始化知识库...")
    sample_knowledge = create_sample_knowledge()
    
    print("[2/4] 构建向量数据库...")
    vector_db = VectorDatabase()
    vector_db.add_items(sample_knowledge)
    
    print("[3/4] 初始化问答引擎...")
    qa_system = ConstructionQASystem(vector_db)
    
    print("[4/4] 系统就绪！\n")
    
    while True:
        question = input("\n请输入您的问题（输入'quit'退出）：").strip()
        
        if question.lower() in ['quit', '退出', 'q']:
            print("\n感谢使用，再见！")
            break
        
        if not question:
            continue
        
        print("\n正在思考中...")
        result = qa_system.ask(question)
        
        print("\n" + "-" * 60)
        print(f"问题：{result['question']}")
        print(f"问题类型：{result['question_type']}")
        print(f"置信度：{result['confidence']:.2f}")
        print("\n回答：")
        print(result['answer'])
        print(f"\n来源：{', '.join(result['sources'])}")
        print("-" * 60)


if __name__ == "__main__":
    main()

