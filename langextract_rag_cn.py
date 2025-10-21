
"""

模仿：
https://github.com/PromtEngineer/LangExtract-RAG


LangExtract + RAG 演示（中文大众点评评论场景）
结构与原示例保持一致：
- Documents：由 get_sample_documents() 返回静态样本（未 chunk）
- LangExtract：尝试使用 langextract，失败回退到正则抽取；抽取后规范化 metadata
- Vector DB：未真正向量化，仅用 SmartVectorStore 在内存列表中模拟索引
- 检索：基于元数据模糊匹配与子串匹配
- Query -> Filter -> 返回匹配文档
"""


import os
import re
from typing import List, Dict
from dotenv import load_dotenv

# aliyun
from langextract.providers.openai import OpenAILanguageModel

apikey = 'sk-81f45edbfae243c3bfd2d8ff82f034ac'
model= OpenAILanguageModel(
            model_id='qwen-plus',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=apikey
            )



load_dotenv()

def get_sample_documents():
    """返回若干中文大众点评风格的评论样本（静态示例）"""
    return [
        {
            "id": "rev_001",
            "title": "老王烧烤 - 一次愉快的就餐体验",
            "content": "店名：老王烧烤\n评分：5星\n时间：2024-08-12\n评价：味道很棒，羊肉串多汁，环境干净，服务态度热情。价格适中，推荐给朋友。标签：环境好, 服务好, 味道棒"
        },
        {
            "id": "rev_002",
            "title": "小南面馆 - 面条一般，服务较慢",
            "content": "店名：小南面馆\n评分：3星\n时间：2023-11-02\n评价：汤面味道中规中矩，分量偏少。上菜慢，服务员不太热情。人均偏贵。标签：味道一般, 上菜慢, 偏贵"
        },
        {
            "id": "rev_003",
            "title": "绿茶餐厅 - 商务聚餐首选",
            "content": "店名：绿茶餐厅\n评分：4星\n时间：2024-04-05\n评价：环境雅致，适合聚餐。菜品口味稳定，价格略高但服务到位。停车方便。标签：环境好, 适合聚餐, 服务好"
        },
        {
            "id": "rev_004",
            "title": "海鲜一品 - 食材新鲜但价格高",
            "content": "店名：海鲜一品\n评分：2星\n时间：2024-06-20\n评价：海鲜确实新鲜，但份量少且价格明显偏高。服务态度一般，有点失望。标签：食材新鲜, 偏贵, 服务一般"
        }
    ]


class FixedLangExtractProcessor:
    """面向中文点评的元数据抽取器；优先使用 langextract（若存在），否则使用正则回退"""

    def __init__(self):
        try:
            import langextract as lx
            self.lx = lx
            self.setup_complete = True
            print("✅ LangExtract 已初始化（中文模式）")
        except ImportError:
            print("⚠️  未安装 langextract，使用正则回退逻辑")
            self.setup_complete = False

    def extract_metadata(self, documents: List[Dict]) -> List[Dict]:
        """对多个文档抽取并规范化 metadata"""
        if not self.setup_complete:
            return self._enhanced_regex_extraction(documents)

        # 这里给 langextract 的 prompt（中文描述）
        prompt = """
        从中文餐厅评论中提取以下字段：
        1. shop_name: 餐厅名称
        2. rating: 评分，仅返回数字，比如 "5" 或 "3"
        3. review_date: 评论日期，格式 YYYY-MM-DD（如能抽取）
        4. review_focus: 评论关注点，必须从：'口味', '环境', '服务', '价格' 中选择最主要的一项
        5. tags: 评论中的标签列表（如 环境好，服务好）
        6. sentiment: 整体情感，'positive'/'negative'/'neutral'
        请尽量精确，只输出字段值，不要额外解释。
        """

        examples = [
            self.lx.data.ExampleData(
                text="店名：示例店\n评分：5星\n时间：2024-01-01\n评价：味道很好，服务热情。标签：味道好, 服务好",
                extractions=[
                    self.lx.data.Extraction(extraction_class="shop_name", extraction_text="示例店", attributes={}),
                    self.lx.data.Extraction(extraction_class="rating", extraction_text="5", attributes={}),
                    self.lx.data.Extraction(extraction_class="review_date", extraction_text="2024-01-01", attributes={}),
                    self.lx.data.Extraction(extraction_class="review_focus", extraction_text="口味", attributes={}),
                    self.lx.data.Extraction(extraction_class="tags", extraction_text="味道好, 服务好", attributes={}),
                    self.lx.data.Extraction(extraction_class="sentiment", extraction_text="positive", attributes={}),
                ]
            )
        ]

        extracted_docs = []
        for doc in documents:
            print(f"📄 处理文档: {doc['title']}")
            try:
                result = self.lx.extract(
                    text_or_documents=doc['content'],
                    prompt_description=prompt,
                    examples=examples,
                    model= model,
                    extraction_passes=2
                )
                metadata = self._process_and_normalize(result.extractions, doc)
            except Exception as e:
                print(f"  ⚠️ LangExtract 抽取失败: {e}")
                metadata = self._enhanced_regex_extraction([doc])[0]['metadata']

            extracted_docs.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'metadata': metadata
            })
        return extracted_docs

    def _process_and_normalize(self, extractions, doc: Dict) -> Dict:
        """处理 langextract 的抽取结果并做规范化"""
        metadata = {
            'shop': '未知',
            'rating': 'unknown',
            'date': '',
            'focus': '口味',
            'tags': [],
            'sentiment': 'neutral'
        }

        for extraction in extractions:
            cls = extraction.extraction_class
            txt = extraction.extraction_text.strip() if hasattr(extraction, 'extraction_text') else ''
            if cls == "shop_name":
                metadata['shop'] = txt
            elif cls == "rating":
                # 只保留数字部分
                m = re.search(r'(\d)', txt)
                if m:
                    metadata['rating'] = m.group(1)
                else:
                    metadata['rating'] = txt
            elif cls == "review_date":
                metadata['date'] = txt
            elif cls == "review_focus":
                metadata['focus'] = txt
            elif cls == "tags":
                # 可能是逗号分隔
                tags = [t.strip() for t in re.split(r'[，,；;]', txt) if t.strip()]
                metadata['tags'] = tags
            elif cls == "sentiment":
                metadata['sentiment'] = txt

        # 回退：若关键字段缺失，使用正则回退
        if metadata['shop'] == '未知' or metadata['rating'] == 'unknown':
            regex_meta = self._enhanced_regex_extraction([doc])[0]['metadata']
            if metadata['shop'] == '未知':
                metadata['shop'] = regex_meta['shop']
            if metadata['rating'] == 'unknown':
                metadata['rating'] = regex_meta['rating']
            if not metadata['tags']:
                metadata['tags'] = regex_meta['tags']

        return metadata

    def _enhanced_regex_extraction(self, documents: List[Dict]) -> List[Dict]:
        """针对中文点评的正则抽取逻辑"""
        extracted_docs = []
        for doc in documents:
            metadata = {
                'shop': '未知',
                'rating': 'unknown',
                'date': '',
                'focus': '口味',
                'tags': [],
                'sentiment': 'neutral'
            }
            title = doc.get('title', '')
            content = doc.get('content', '')

            # 店名优先从 content 中的 "店名：XXX" 提取，其次尝试从 title 提取 "店名 - 描述"
            shop_match = re.search(r'店名[:：]\s*([^\n]+)', content)
            if shop_match:
                metadata['shop'] = shop_match.group(1).strip()
            else:
                # title 中可能是 "店名 - 描述"
                title_match = re.match(r'([\u4e00-\u9fff\w\s]+)\s*[-–—]\s*', title)
                if title_match:
                    metadata['shop'] = title_match.group(1).strip()

            # 评分：寻找 "5星", "评分：5" 或单独数字（1-5）
            rating_match = re.search(r'(\d)\s*星|评分[:：]\s*(\d)', content)
            if rating_match:
                metadata['rating'] = rating_match.group(1) if rating_match.group(1) else rating_match.group(2)
            else:
                # 尝试在 title 中找
                r2 = re.search(r'(\d)\s*星', title)
                if r2:
                    metadata['rating'] = r2.group(1)

            # 日期：简单匹配 YYYY-MM-DD
            date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', content)
            if date_match:
                metadata['date'] = date_match.group(1)

            # tags：寻找 "标签：" 或句中常见短语
            tags_match = re.search(r'标签[:：]\s*([^\n]+)', content)
            if tags_match:
                tags = [t.strip() for t in re.split(r'[，,；;]', tags_match.group(1)) if t.strip()]
                metadata['tags'] = tags
            else:
                # 简单关键词识别作为 tags
                possible_tags = []
                for kw in ['环境好', '服务好', '味道棒', '味道一般', '偏贵', '食材新鲜', '上菜慢', '适合聚餐']:
                    if kw in content:
                        possible_tags.append(kw)
                metadata['tags'] = possible_tags

            # 关注点（focus）：根据关键词判断
            if any(k in content for k in ['味', '口味', '好吃', '难吃']):
                metadata['focus'] = '口味'
            elif any(k in content for k in ['环境', '雅致', '干净', '嘈杂']):
                metadata['focus'] = '环境'
            elif any(k in content for k in ['服务', '上菜', '态度']):
                metadata['focus'] = '服务'
            elif any(k in content for k in ['价格', '偏贵', '便宜', '人均']):
                metadata['focus'] = '价格'

            # 情感判定（粗略）
            if any(p in content for p in ['很好', '棒', '推荐', '满意', '愉快', '喜欢']):
                metadata['sentiment'] = 'positive'
            elif any(n in content for n in ['差', '失望', '不满', '不好', '太贵', '一般']):
                metadata['sentiment'] = 'negative'
            else:
                metadata['sentiment'] = 'neutral'

            extracted_docs.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'metadata': metadata
            })
        return extracted_docs


class SmartVectorStore:
    """内存级别的“智能”索引：基于元数据的模糊匹配 + 文本子串匹配（未做向量化）"""

    def __init__(self):
        self.documents = []

    def add_documents(self, docs: List[Dict]):
        self.documents = docs
        print(f"✅ 已索引 {len(docs)} 条评论")

    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """基于 query 的简单检索；若提供 filters 则做元数据过滤"""
        if not filters:
            return [doc for doc in self.documents if any(word.lower() in doc['content'].lower() for word in query.split())]

        filtered_docs = []
        for doc in self.documents:
            match = True
            md = doc.get('metadata', {})

            # 店铺模糊匹配（支持部分关键词匹配）
            if 'shop' in filters:
                q_shop = filters['shop'].lower()
                doc_shop = md.get('shop', '未知').lower()
                if q_shop not in doc_shop and doc_shop not in q_shop:
                    q_keywords = set(re.sub(r'(店|餐厅|馆|酒楼|烧烤|面馆)', '', q_shop).split())
                    doc_keywords = set(re.sub(r'(店|餐厅|馆|酒楼|烧烤|面馆)', '', doc_shop).split())
                    if not q_keywords.intersection(doc_keywords):
                        match = False

            # 评分精确匹配或大于等于
            if 'rating' in filters:
                try:
                    target = int(filters['rating'])
                    try:
                        doc_rating = int(md.get('rating', '0'))
                    except:
                        doc_rating = 0
                    if doc_rating < target:
                        match = False
                except:
                    # 非数字则做包含匹配
                    if filters['rating'] != md.get('rating'):
                        match = False

            # 关注点匹配（exact）
            if 'focus' in filters:
                if filters['focus'] != md.get('focus'):
                    match = False

            # 情感过滤（positive/negative/neutral）
            if 'sentiment' in filters:
                if filters['sentiment'] != md.get('sentiment'):
                    match = False

            if match:
                # 最后再做一次内容关键词匹配，确保与 query 语义相关（基于子串）
                if any(word.lower() in doc['content'].lower() for word in query.split()):
                    filtered_docs.append(doc)

        return filtered_docs


def extract_smart_filters(query: str) -> Dict:
    """从中文查询中抽取过滤条件（如店名、评分、关注点、情感）"""
    filters = {}
    q = query.lower()

    # 提取评分：如 "5星"、"评分5"、"至少4分" 等
    m = re.search(r'至少\s*(\d)\s*|(\d)\s*星|评分[:：]?\s*(\d)', q)
    if m:
        for g in m.groups():
            if g:
                filters['rating'] = g
                break

    # 店名简单匹配：若查询中包含明显的店名关键词（示例中包括老王、小南、绿茶、海鲜）
    if '老王' in q or '老王烧烤' in q:
        filters['shop'] = '老王烧烤'
    elif '小南' in q or '小南面馆' in q:
        filters['shop'] = '小南面馆'
    elif '绿茶' in q or '绿茶餐厅' in q:
        filters['shop'] = '绿茶餐厅'
    elif '海鲜' in q or '海鲜一品' in q:
        filters['shop'] = '海鲜一品'

    # 关注点识别
    if any(k in q for k in ['口味', '味道', '好吃', '难吃']):
        filters['focus'] = '口味'
    elif any(k in q for k in ['环境', '干净', '雅致', '嘈杂']):
        filters['focus'] = '环境'
    elif any(k in q for k in ['服务', '上菜', '态度']):
        filters['focus'] = '服务'
    elif any(k in q for k in ['价格', '贵', '便宜', '人均']):
        filters['focus'] = '价格'

    # 情感识别（好评/差评）
    if any(p in q for p in ['好评', '推荐', '满意', '喜欢']):
        filters['sentiment'] = 'positive'
    elif any(n in q for n in ['差评', '失望', '不满', '差']):
        filters['sentiment'] = 'negative'

    return filters


def main():
    print("=== 中文大众点评评论 RAG 演示 ===")

    # Step 1: 加载文档
    print("📚 正在加载样本评论...")
    documents = get_sample_documents()

    # Step 2: 抽取 metadata
    print("\n🔍 使用增强抽取系统提取元数据...")
    extractor = FixedLangExtractProcessor()
    extracted_docs = extractor.extract_metadata(documents)

    # 显示抽取结果
    print("\n📊 抽取并规范化的元数据：")
    for doc in extracted_docs:
        md = doc['metadata']
        print(f"\n  {doc['id']} ({doc['title']}):")
        print(f"    店名: '{md['shop']}'")
        print(f"    评分: '{md['rating']}'")
        print(f"    关注点: '{md['focus']}'")
        print(f"    情感: '{md['sentiment']}'")
        if md.get('tags'):
            print(f"    标签: {md['tags']}")

    # Step 3: 索引到 SmartVectorStore（内存）
    print("\n💾 索引文档到 SmartVectorStore（模拟）...")
    vector_store = SmartVectorStore()
    vector_store.add_documents(extracted_docs)

    # Step 4: 测试查询
    test_queries = [
        "如何评价老王的味道？",
        "有哪些 5星 的推荐？",
        "关于上菜慢的差评有哪些？",
        "绿茶餐厅 环境 怎么样？",
        "海鲜 一品 是否 偏贵？"
    ]

    print("\n🔬 测试检索：")
    print("=" * 70)
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        filters = extract_smart_filters(query)
        if filters:
            print(f"   🎯 抽取到的过滤条件: {filters}")

        with_results = vector_store.search(query, filters)
        print(f"   ✅ 使用元数据过滤检索到: {len(with_results)} 条评论")
        if with_results:
            for r in with_results:
                md = r['metadata']
                print(f"      - {r['id']}: {md['shop']} {md['rating']}星 （关注点: {md['focus']}，情感: {md['sentiment']})")
        print("\n 实际返回文档: ", with_results)

        without_results = vector_store.search(query, None)
        print(f"\n ❌ 不使用过滤条件检索到: {len(without_results)} 条评论")
        print("\n 实际返回文档: ", without_results)


if __name__ == "__main__":
    main()

