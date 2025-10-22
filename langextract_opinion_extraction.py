"""
增强版情感观点三元组抽取器（v7）

说明：
- 将 Qwen/OpenAI 兼容模型的构建移动到 __init__ 中，构建后的模型实例保存在 self.model（可为 None）。
- 构造时仍然要求提供 qwen_apikey（示例：'sk-xxxx'），以便尽量在初始化阶段就准备好远程模型。
- 如果 langextract 或 OpenAILanguageModel 提供器缺失，依然支持回退启发式抽取。
- 注释全部为中文，代码结构较 v6 更加简洁明确。

"""

import os
import json
import re
from typing import List, Dict, Optional

# 尝试导入 langextract（优先使用）
try:
    import langextract as lx  # type: ignore
    LANGEXTRACT_AVAILABLE = True
except Exception:
    lx = None
    LANGEXTRACT_AVAILABLE = False

# 尝试导入 OpenAILanguageModel（用于 Qwen 兼容调用）
try:
    from langextract.providers.openai import OpenAILanguageModel  # type: ignore
    OPENAI_LM_AVAILABLE = True
except Exception:
    OpenAILanguageModel = None  # type: ignore
    OPENAI_LM_AVAILABLE = False

# 子维度关键词字典（回退使用）
SUBASPECT_KEYWORDS = {
    "位置": {
        "交通是否便利": ["地铁", "公交", "交通", "步行", "接驳", "出站"],
        "距离商圈远近": ["商圈", "远", "离商圈", "距离"],
        "是否容易寻找": ["门牌", "巷子", "不好找", "入口", "招牌"]
    },
    "服务": {
        "排队等候时间": ["排队", "等位", "等待", "候位"],
        "服务人员态度": ["服务员", "态度", "热情", "冷淡", "礼貌"],
        "是否容易停车": ["停车", "停车位", "泊车"],
        "点菜/上菜速度": ["上菜", "上桌", "上菜速度", "上菜慢"]
    },
    "价格": {
        "价格水平": ["价格", "价位", "贵", "便宜"],
        "性价比": ["性价比", "划算", "不划算"],
        "折扣力度": ["折扣", "满减", "优惠", "活动"]
    },
    "环境": {
        "装修情况": ["装修", "风格", "布置", "设计"],
        "嘈杂情况": ["嘈杂", "安静", "吵", "噪音", "音乐"],
        "就餐空间": ["座位", "空间", "拥挤", "桌间距"],
        "卫生情况": ["卫生", "干净", "脏", "油渍", "异味"]
    },
    "菜品": {
        "分量": ["分量", "份量", "量", "很足", "太少"],
        "口感": ["口感", "鲜嫩", "多汁", "偏咸", "味道"],
        "外观": ["外观", "颜值", "摆盘"],
        "推荐程度": ["推荐", "强烈推荐", "值得"]
    },
    "其他": {
        "本次消费感受": ["体验", "感受", "体验感"],
        "再次消费的意愿": ["还会来", "不会再来", "会再来", "回访"]
    }
}

# 情感词与否定词（回退使用）
POS_WORDS = ["好", "棒", "赞", "推荐", "满意", "喜欢", "不错", "鲜美", "到位", "超赞", "惊喜", "贴心", "热情"]
NEG_WORDS = ["差", "失望", "不满", "不好", "太咸", "糟糕", "没味", "坑", "不推荐", "昂贵", "拥挤", "脏"]
NEGATION_WORDS = ["不", "没", "无", "没有", "并不", "并非", "非"]
DEGREE_WORDS = ["很", "非常", "极其", "特别", "十分"]


class EnhancedOpinionExtractorV7:
    """
    情感观点三元组抽取器（v7）

    主要变化：
    - 在 __init__ 中尝试构建 Qwen 兼容的模型实例，并将其放入 self.model。
    - 提供简洁的 langextract 调用封装 _call_langextract，解析函数 _parse_extractions，回退函数 _fallback_extract。
    - extract_triples 使用 self.model（若存在且 use_qwen_model=True）进行抽取，并打印每条文本使用了模型还是回退规则。
    """

    def __init__(self, qwen_apikey: str,model_id = 'qwen-turbo'):
        """
        初始化抽取器，必须传入 qwen_apikey（示例：'sk-xxxx'）。
        在构造函数中尝试构建 Qwen 模型并保存到 self.model；若无法构建，self.model 为 None。
        """
        if not qwen_apikey or not isinstance(qwen_apikey, str):
            raise ValueError("必须提供 qwen_apikey，示例：'sk-xxxx'")

        # 保存 qwen api key
        self.qwen_apikey = qwen_apikey

        # 初始化回退资源（无论是否安装 langextract 都需要）
        self.subaspect_keywords = SUBASPECT_KEYWORDS
        self.pos_words = POS_WORDS
        self.neg_words = NEG_WORDS
        self.negation_words = NEGATION_WORDS
        self.degree_words = DEGREE_WORDS

        # langextract 是否可用（仅表示库可用，模型是否传入另当别论）
        self.use_langextract = LANGEXTRACT_AVAILABLE

        # 如果 langextract 可用，准备示例和 prompt（示例用于模型抽取时参考）
        if self.use_langextract:
            self.lx = lx
            try:
                self.examples = self._build_examples()
            except Exception:
                self.examples = None
            self.prompt = self._build_prompt()
        else:
            self.lx = None
            self.examples = None
            self.prompt = self._build_prompt()

        # 在初始化阶段尝试构建 Qwen/OpenAI 兼容模型并赋值给 self.model
        self.model = None
        if OPENAI_LM_AVAILABLE:
            try:
                self.model = self._build_qwen_model(apikey=self.qwen_apikey,model_id = model_id)
                print("✅ 已在 __init__ 中构建 Qwen/OpenAI 兼容模型，并保存到 self.model")
            except Exception as e:
                self.model = None
                print(f"⚠️ 在 __init__ 构建 Qwen 模型失败：{e}，后续抽取将回退或使用 langextract 默认模型")
        else:
            print("⚠️ 未检测到 langextract.providers.openai.OpenAILanguageModel，无法构建 Qwen 模型（self.model=None）")

    # 以下为内部方法：构建 prompt、构建 examples、构建 model、调用模型、解析结果与回退抽取
    def _build_prompt(self) -> str:
        """构建传给 langextract 的中文提示语，要求返回 Extraction 列表"""
        return """
        任务：从中文餐厅评论中抽取情感观点三元组。
        每个三元组包含字段：
          - aspect（类别）：位置/服务/价格/环境/菜品/其他
          - sub_aspect（子维度，可选）：例如 交通是否便利、上菜速度 等
          - opinion（观点短语）：一句话或短语
          - sentiment（情感）：positive/negative/neutral

        严格的输出要求（必须遵守）：
        - 只输出一个 JSON 代码块，不要任何多余说明或文本。
        - JSON 必须是一个对象，顶层有键 "extractions"，其值是一个数组。
        - 数组中每个元素是一个对象，包含：
            - "extraction_class": 固定值 "opinion_triple"
            - "extraction_text": 一个字符串（该字符串本身是 JSON），里面有 aspect、sub_aspect、opinion、sentiment 字段

        示例（模型输出必须与此结构严格一致，包括代码块标记）：
        ```json
        {"extractions":[
          {"extraction_class":"opinion_triple","extraction_text":"{\"aspect\":\"菜品\",\"sub_aspect\":\"口感\",\"opinion\":\"羊肉串多汁\",\"sentiment\":\"positive\"}"},
          {"extraction_class":"opinion_triple","extraction_text":"{\"aspect\":\"环境\",\"sub_aspect\":\"噪音\",\"opinion\":\"太吵了\",\"sentiment\":\"negative\"}"}
        ]}
        ```
        仅返回上面的 JSON 代码块，不要额外的文字或解释。 
        """

    def _build_examples(self) -> List:
        """
        扩展后的 ExampleData 集合，覆盖每个点评分类和子维度（每个子维度至少 2 个示例）。
        返回 langextract.data.ExampleData 的列表（若 langextract 可用）。
        """
        Data = self.lx.data
        exs = []

        def add_example(text: str, triples: List[Dict]):
            extractions = []
            for t in triples:
                extractions.append(Data.Extraction(
                    extraction_class="opinion_triple",
                    extraction_text=json.dumps(t, ensure_ascii=False),
                    attributes={}
                ))
            exs.append(Data.ExampleData(text=text, extractions=extractions))

        # ----------------- 位置 子维度示例 -----------------
        add_example(
            "店名：老王烧烤\n评价：地理位置很好，离地铁站步行5分钟，门口有停车位，周边公交也方便。",
            [{"aspect":"位置","sub_aspect":"交通是否便利","opinion":"离地铁站步行5分钟","sentiment":"positive"},
             {"aspect":"位置","sub_aspect":"是否容易停车","opinion":"门口有停车位","sentiment":"positive"}]
        )
        add_example(
            "店名：远郊家常菜\n评价：离市中心较远，需要打车大约半小时，周边交通不太方便。",
            [{"aspect":"位置","sub_aspect":"距离商圈远近","opinion":"离市中心较远，打车约30分钟","sentiment":"negative"}]
        )
        add_example(
            "店名：巷里小馆\n评价：门牌不显眼，店铺在小巷里，第一次来花了不少时间才找到入口。",
            [{"aspect":"位置","sub_aspect":"是否容易寻找","opinion":"门牌不显眼，巷子里难找","sentiment":"negative"}]
        )
        add_example(
            "店名：中心食府\n评价：位于商圈中心，出行很方便，附近购物餐饮一应俱全。",
            [{"aspect":"位置","sub_aspect":"距离商圈远近","opinion":"位于商圈中心，交通便利","sentiment":"positive"}]
        )
        add_example(
            "店名：临街小店\n评价：虽然靠近地铁，但门口车位较少，午高峰取车不便。",
            [{"aspect":"位置","sub_aspect":"是否容易停车","opinion":"门口车位少，午高峰取车不便","sentiment":"negative"}]
        )

        # ----------------- 服务 子维度示例 -----------------
        add_example(
            "店名：绿茶餐厅\n评价：服务员非常热情，上菜速度也很快，服务细致周到。",
            [{"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务员非常热情","sentiment":"positive"},
             {"aspect":"服务","sub_aspect":"点菜/上菜速度","opinion":"上菜速度很快","sentiment":"positive"}]
        )
        add_example(
            "店名：海鲜一品\n评价：周末我们等了近四十分钟才有位，等位时间太久体验很差。",
            [{"aspect":"服务","sub_aspect":"排队等候时间","opinion":"等位约40分钟","sentiment":"negative"}]
        )
        add_example(
            "店名：大道停车馆\n评价：餐厅自带地下停车场，停车非常方便，不用再东找车位。",
            [{"aspect":"服务","sub_aspect":"是否容易停车","opinion":"自带地下停车场，停车方便","sentiment":"positive"}]
        )
        add_example(
            "店名：慢味居\n评价：上菜速度慢，经常需要催菜，适合不着急的用餐场景。",
            [{"aspect":"服务","sub_aspect":"点菜/上菜速度","opinion":"上菜慢，需要催菜","sentiment":"negative"}]
        )
        add_example(
            "店名：笑脸小馆\n评价：服务员态度冷淡，点单时没有推荐，整体服务体验一般。",
            [{"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务员态度冷淡","sentiment":"negative"}]
        )

        # ----------------- 价格 子维度示例 -----------------
        add_example(
            "店名：陈记小吃\n评价：人均不到30元，份量还不错，性价比很高，学生党非常适合。",
            [{"aspect":"价格","sub_aspect":"价格水平","opinion":"人均不到30元","sentiment":"positive"},
             {"aspect":"价格","sub_aspect":"性价比","opinion":"性价比高，份量足","sentiment":"positive"}]
        )
        add_example(
            "店名：海鲜一品\n评价：虽然食材不错，但人均偏高，吃下来预算超出预期，不太划算。",
            [{"aspect":"价格","sub_aspect":"价格水平","opinion":"人均偏高","sentiment":"negative"},
             {"aspect":"价格","sub_aspect":"性价比","opinion":"性价比不高","sentiment":"negative"}]
        )
        add_example(
            "店名：优惠小馆\n评价：经常有满减和折扣，活动期间非常划算，推荐关注官方活动。",
            [{"aspect":"价格","sub_aspect":"折扣力度","opinion":"满减促销力度大，折扣多","sentiment":"positive"}]
        )
        add_example(
            "店名：高端料理\n评价：价格较贵，但食材与体验匹配，适合节日或特殊场合消费。",
            [{"aspect":"价格","sub_aspect":"价格水平","opinion":"价格较贵但体验相匹配","sentiment":"neutral"}]
        )

        # ----------------- 环境 子维度示例 -----------------
        add_example(
            "店名：玫瑰咖啡\n评价：装修风格文艺，光线柔和，座位舒适，很适合拍照和聊天。",
            [{"aspect":"环境","sub_aspect":"装修情况","opinion":"装修文艺、光线好","sentiment":"positive"},
             {"aspect":"环境","sub_aspect":"就餐空间","opinion":"座位舒适","sentiment":"positive"}]
        )
        add_example(
            "店名：嘈杂烧烤\n评价：餐厅里音乐很大，整体比较嘈杂，不适合安静聊天。",
            [{"aspect":"环境","sub_aspect":"嘈杂情况","opinion":"音乐较大、环境嘈杂","sentiment":"negative"}]
        )
        add_example(
            "店名：脏碗馆\n评价：桌面有油渍，餐具不够干净，卫生需要改进，影响食欲。",
            [{"aspect":"环境","sub_aspect":"卫生情况","opinion":"桌面有油渍，餐具不干净","sentiment":"negative"}]
        )
        add_example(
            "店名：小而精餐厅\n评价：店内空间较小，桌与桌之间距离近，聚会时可能会觉得拥挤。",
            [{"aspect":"环境","sub_aspect":"就餐空间","opinion":"空间较小、桌间距近","sentiment":"negative"}]
        )

        # ----------------- 菜品 子维度示例 -----------------
        add_example(
            "店名：老街烧肉\n评价：牛肉鲜嫩多汁，分量足，食材很新鲜，强烈推荐牛肉类菜品。",
            [{"aspect":"菜品","sub_aspect":"口感","opinion":"牛肉鲜嫩多汁","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"分量","opinion":"分量足","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"推荐程度","opinion":"强烈推荐","sentiment":"positive"}]
        )
        add_example(
            "店名：懒熊披萨\n评价：摆盘很漂亮，但口感偏硬且边缘干，可能需要改进烘烤时间。",
            [{"aspect":"菜品","sub_aspect":"外观","opinion":"摆盘漂亮","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"口感偏硬、边缘干","sentiment":"negative"}]
        )
        add_example(
            "店名：家常米线\n评价：分量稍少，适合胃口小的顾客，口味偏清淡。",
            [{"aspect":"菜品","sub_aspect":"分量","opinion":"分量稍少","sentiment":"neutral"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"口味偏清淡","sentiment":"neutral"}]
        )
        add_example(
            "店名：甜品屋\n评价：甜点外观精致，口感层次丰富，非常值得一试。",
            [{"aspect":"菜品","sub_aspect":"外观","opinion":"外观精致","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"口感层次丰富","sentiment":"positive"}]
        )

        # ----------------- 其他 子维度示例 -----------------
        add_example(
            "店名：素心斋\n评价：整体用餐体验令人满意，会考虑带父母再来一次。",
            [{"aspect":"其他","sub_aspect":"本次消费感受","opinion":"整体体验令人满意","sentiment":"positive"},
             {"aspect":"其他","sub_aspect":"再次消费的意愿","opinion":"会考虑再次消费","sentiment":"positive"}]
        )
        add_example(
            "店名：米线小馆\n评价：总体尚可，但这次体验不算完美，短期内可能不会回访。",
            [{"aspect":"其他","sub_aspect":"本次消费感受","opinion":"总体尚可但不完美","sentiment":"neutral"},
             {"aspect":"其他","sub_aspect":"再次消费的意愿","opinion":"短期内可能不会回访","sentiment":"negative"}]
        )
        add_example(
            "店名：回头客饭庄\n评价：这次体验一般，但若有活动可能会再来，近期不会成为常去地点。",
            [{"aspect":"其他","sub_aspect":"本次消费感受","opinion":"体验一般","sentiment":"neutral"},
             {"aspect":"其他","sub_aspect":"再次消费的意愿","opinion":"可能在活动时再来","sentiment":"neutral"}]
        )

        # ----------------- 复杂/混合 表达示例 -----------------
        add_example(
            "店名：综合馆\n评价：味道不错但价格偏高，服务一般，离地铁站还算近，门口停车位不多。",
            [{"aspect":"菜品","sub_aspect":"口感","opinion":"味道不错","sentiment":"positive"},
             {"aspect":"价格","sub_aspect":"价格水平","opinion":"价格偏高","sentiment":"negative"},
             {"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务一般","sentiment":"neutral"},
             {"aspect":"位置","sub_aspect":"交通是否便利","opinion":"离地铁站较近","sentiment":"positive"},
             {"aspect":"位置","sub_aspect":"是否容易停车","opinion":"门口停车位少","sentiment":"negative"}]
        )
        add_example(
            "店名：混合体验店\n评价：菜品新鲜但上菜慢，店内有点嘈杂，环境和服务都有提升空间，价格中等偏上。",
            [{"aspect":"菜品","sub_aspect":"口感","opinion":"菜品新鲜","sentiment":"positive"},
             {"aspect":"服务","sub_aspect":"点菜/上菜速度","opinion":"上菜慢","sentiment":"negative"},
             {"aspect":"环境","sub_aspect":"嘈杂情况","opinion":"店内较嘈杂","sentiment":"negative"},
             {"aspect":"价格","sub_aspect":"价格水平","opinion":"价格中等偏上","sentiment":"neutral"}]
        )

        # 新增至少 5 条复杂/混合表达示例（覆盖否定、转折、并列、隐含信息、多观点）
        add_example(
            "店名：对比餐厅\n评价：两年前来时服务很好，这次感觉服务下降了不少，但菜品依旧保持水准。位置挺方便但停车不太好，价位有点上调。",
            [{"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务相比以前下降","sentiment":"negative"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"菜品仍然保持水准","sentiment":"positive"},
             {"aspect":"位置","sub_aspect":"是否容易停车","opinion":"停车不太好","sentiment":"negative"},
             {"aspect":"价格","sub_aspect":"价格水平","opinion":"价位上调","sentiment":"negative"}]
        )
        add_example(
            "店名：折扣季餐厅\n评价：平时觉得有点贵，但这次赶上活动满减，性价比瞬间变好；不过上菜速度仍然慢，服务员态度一般。",
            [{"aspect":"价格","sub_aspect":"折扣力度","opinion":"活动满减后性价比变好","sentiment":"positive"},
             {"aspect":"服务","sub_aspect":"点菜/上菜速度","opinion":"上菜速度慢","sentiment":"negative"},
             {"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务员态度一般","sentiment":"neutral"}]
        )
        add_example(
            "店名：矛盾店\n评价：环境很棒，适合拍照，但菜品口味太重且偏咸，价格也不便宜，适合拍照不适合天天吃。",
            [{"aspect":"环境","sub_aspect":"装修情况","opinion":"环境很棒，适合拍照","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"口味太重且偏咸","sentiment":"negative"},
             {"aspect":"价格","sub_aspect":"价格水平","opinion":"价格不便宜","sentiment":"negative"},
             {"aspect":"其他","sub_aspect":"本次消费感受","opinion":"适合拍照但不适合天天吃","sentiment":"neutral"}]
        )
        add_example(
            "店名：夜宵馆\n评价：夜里来吃很方便，味道也不错，但店里比较吵，座位紧凑，二三人聚餐可以，家人聚会就不太合适。",
            [{"aspect":"位置","sub_aspect":"交通是否便利","opinion":"夜里来吃很方便","sentiment":"positive"},
             {"aspect":"菜品","sub_aspect":"口感","opinion":"味道不错","sentiment":"positive"},
             {"aspect":"环境","sub_aspect":"嘈杂情况","opinion":"店里比较吵","sentiment":"negative"},
             {"aspect":"环境","sub_aspect":"就餐空间","opinion":"座位紧凑，不适合大家庭聚会","sentiment":"negative"}]
        )
        add_example(
            "店名：家庭聚餐店\n评价：菜量很足，适合多人分食，服务也比较周到，不过停车位紧张，周末最好预留充足时间。",
            [{"aspect":"菜品","sub_aspect":"分量","opinion":"菜量很足，适合多人分食","sentiment":"positive"},
             {"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务比较周到","sentiment":"positive"},
             {"aspect":"位置","sub_aspect":"是否容易停车","opinion":"停车位紧张，周末需预留时间","sentiment":"negative"}]
        )
        add_example(
            "店名：口碑两极店\n评价：朋友们有人觉得味道一绝，也有人觉得太油腻，这家店真是看人选菜，服务倒是没什么大问题。",
            [{"aspect":"菜品","sub_aspect":"口感","opinion":"有人觉得味道很棒，有人觉得太油腻","sentiment":"neutral"},
             {"aspect":"服务","sub_aspect":"服务人员态度","opinion":"服务没有明显问题","sentiment":"neutral"}]
        )

        # 返回完整示例集合
        return exs

    def _build_qwen_model(self, apikey: Optional[str] = None,model_id = "qwen-turbo"):
        """
        构建 Qwen/OpenAI 兼容的模型实例，返回可传入 lx.extract 的 model 对象。
        若无法构建则抛出异常。
        """
        if not OPENAI_LM_AVAILABLE:
            raise RuntimeError("缺少 OpenAILanguageModel 提供器，请安装 langextract.providers.openai")
        key = apikey or self.qwen_apikey
        if not key:
            raise RuntimeError("未提供 qwen_apikey")
        model = OpenAILanguageModel(
            model_id=os.getenv("QWEN_MODEL_ID", model_id),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=key
        )
        return model

    def _call_langextract(self, content: str, model=None, extraction_passes: int = 2):
        """
        对单条文本调用 langextract.extract，传入准备好的 prompt、examples 与可选 model。
        返回 langextract 的结果对象；若调用失败会抛出异常，由上层处理回退。
        """
        if not self.use_langextract:
            raise RuntimeError("当前环境未安装 langextract")
        res = self.lx.extract(
            text_or_documents=content,
            prompt_description=self.prompt,
            examples=self.examples,
            model=model,
            extraction_passes=extraction_passes
        )
        return res

    def _parse_extractions(self, extractions) -> List[Dict]:
        """
        解析 langextract 返回的 Extraction 列表，将 extraction_text 中的 JSON 转为标准三元组字典。
        若 sub_aspect 缺失，尝试从 opinion 文本中推断并填充。
        """
        parsed = []
        for ex in extractions:
            try:
                if ex.extraction_class != "opinion_triple":
                    txt = ex.extraction_text.strip()
                    data = json.loads(txt)
                    arr = data if isinstance(data, list) else [data]
                    for it in arr:
                        aspect = it.get("aspect") or it.get("类别") or ""
                        sub = it.get("sub_aspect") or it.get("subAspect") or it.get("细分") or ""
                        opinion = it.get("opinion") or it.get("观点") or ""
                        sentiment = it.get("sentiment") or it.get("情感") or ""
                        if not sub and opinion:
                            inferred = self._infer_subaspect_from_text(opinion, aspect)
                            if inferred:
                                sub = inferred
                        parsed.append({
                            "aspect": aspect,
                            "sub_aspect": sub,
                            "opinion": opinion,
                            "sentiment": sentiment
                        })
            except Exception:
                continue
        return parsed

    def _infer_subaspect_from_text(self, text: str, aspect_hint: Optional[str] = None) -> str:
        """
        基于子维度关键词字典尝试从文本推断出最可能的 sub_aspect。
        优先在给定的 aspect_hint 下查找，其次全局匹配。
        """
        if not text:
            return ""
        candidates = []
        if aspect_hint and aspect_hint in self.subaspect_keywords:
            submap = self.subaspect_keywords[aspect_hint]
            for sub, kws in submap.items():
                cnt = sum(1 for kw in kws if kw in text)
                if cnt > 0:
                    candidates.append((sub, cnt))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        for asp, submap in self.subaspect_keywords.items():
            for sub, kws in submap.items():
                cnt = sum(1 for kw in kws if kw in text)
                if cnt > 0:
                    candidates.append((sub, cnt))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        return ""

    def _fallback_extract(self, content: str) -> List[Dict]:
        """
        回退启发式抽取：
        - 将文本按句拆分
        - 基于子维度关键词检测所属大类
        - 基于词典进行情感打分（包含否定处理与程度词放大）
        - 尝试推断 sub_aspect 并截取关键词附近短片段作为 opinion
        """
        triples = []
        sents = re.split(r'[。\n！？!?]+', content)
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            found_aspects = set()
            for asp, submap in self.subaspect_keywords.items():
                for sub, subkws in submap.items():
                    for kw in subkws:
                        if kw in sent:
                            found_aspects.add(asp)
                            break
            if not found_aspects:
                continue

            pos = sum(1 for w in self.pos_words if w in sent)
            neg = sum(1 for w in self.neg_words if w in sent)

            def neg_near(word):
                idx = sent.find(word)
                if idx == -1:
                    return False
                start = max(0, idx - 3)
                return any(nw in sent[start:idx] for nw in self.negation_words)

            for pw in self.pos_words:
                if pw in sent and neg_near(pw):
                    pos -= 1
            for nw in self.neg_words:
                if nw in sent and neg_near(nw):
                    neg -= 1

            if any(d in sent for d in self.degree_words):
                pos *= 2

            if pos > neg:
                sentiment = "positive"
            elif neg > pos:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            for asp in found_aspects:
                sub = self._infer_subaspect_from_text(sent, asp)
                opinion = sent
                first_kw = None
                for subkws in self.subaspect_keywords.get(asp, {}).values():
                    for kw in subkws:
                        if kw in sent:
                            first_kw = kw
                            break
                    if first_kw:
                        break
                if first_kw:
                    idx = sent.find(first_kw)
                    start = max(0, idx - 12)
                    end = min(len(sent), idx + len(first_kw) + 12)
                    opinion = sent[start:end].strip()
                triples.append({
                    "aspect": asp,
                    "sub_aspect": sub or "",
                    "opinion": opinion,
                    "sentiment": sentiment
                })
        return triples

    def extract_triples(self, documents: List[Dict], use_qwen_model: bool = True) -> List[Dict]:
        """
        对文档列表执行三元组抽取。
        - use_qwen_model=True 时，如果 self.model 存在则把 self.model 传入 langextract（否则使用 langextract 默认模型或回退）
        - 返回每条文档：{"id": doc_id, "triples": [...], "used_model": True/False}
        - 同时在控制台打印每条文本使用模型还是回退规则
        """
        results = []
        for doc in documents:
            used_model_flag = False
            triples = []
            if self.use_langextract:
                try:
                    # 若用户希望使用 qwen 模型且 self.model 已构建，则传入 self.model
                    model_to_use = self.model if use_qwen_model and self.model else None
                    res = self._call_langextract(doc["content"], model=model_to_use, extraction_passes=2)
                    triples = self._parse_extractions(res.extractions)
                    used_model_flag = True if model_to_use else False
                except Exception as e:
                    # 调用失败则使用回退逻辑
                    triples = self._fallback_extract(doc.get("content", ""))
                    used_model_flag = False
                    print("调用失败则使用回退逻辑",e)
            else:
                # langextract 不可用，全部使用回退
                triples = self._fallback_extract(doc.get("content", ""))
                used_model_flag = False

            # 打印每条文档的抽取方式
            method_desc = "使用大模型(通过 langextract)" if used_model_flag else "使用回退规则"
            print(f"文档 {doc.get('id')} 抽取方式：{method_desc}")
            results.append({"id": doc.get("id"), "triples": triples, "used_model": used_model_flag})
        return results


# -------------------------
# 测试用示例文档（至少 5 篇，文本较长且包含多方面观点）
# -------------------------
def build_test_documents() -> List[Dict]:
    docs = [
        {
            "id": "t1",
            "content": "店名：老王烧烤\n评分：5星\n时间：2024-08-12\n评价：我们一家四口晚上去吃，羊肉串多汁非常好吃，老板还推荐了几款特色蘸料，味道很到位。服务员态度热情，上菜也很快。餐厅门口就有停车位，离地铁站步行约5分钟，非常方便。总体很满意，会再来。标签：味道棒, 服务好"
        },
        {
            "id": "t2",
            "content": "店名：海鲜一品\n评分：2星\n时间：2024-06-20\n评价：这次经历比较糟糕。我们点的海鲜分量少且价格偏高，味道也不如预期。服务员态度冷淡，上菜慢，结账时还出现账单错误。桌面有油渍，整体卫生情况不佳。门口停车位少，周末更难停。不会再来了。标签：偏贵, 卫生一般"
        },
        {
            "id": "t3",
            "content": "店名：绿茶餐厅\n评分：4星\n时间：2024-04-05\n评价：下午来吃了个轻食，环境雅致安静，适合谈话。甜点和咖啡口感都不错，分量适中。服务态度总体良好，但晚高峰时上菜速度有点慢。位置靠近商圈，但门牌不太显眼，第一次来找了好一会儿。总体还会考虑再来。标签：环境好, 适合聚餐"
        },
        {
            "id": "t4",
            "content": "店名：小南面馆\n评分：3星\n时间：2023-11-02\n评价：面条口味中规中矩，汤有些清淡。上菜速度慢，周末高峰期排队比较长。人均偏高，性价比一般。位置离地铁站有段距离，公交也不太方便。服务员态度一般。标签：味道一般, 上菜慢"
        },
        {
            "id": "t5",
            "content": "店名：玫瑰咖啡\n评分：5星\n时间：2024-07-21\n评价：这家咖啡店装修很有格调，光线好，适合拍照和聊天。甜点精致，份量适中。服务员推荐的拿铁很不错，店内也比较安静。稍微不方便的是座位较少，高峰期需要排队。总体体验非常好，会推荐朋友来。标签：环境好, 甜点推荐"
        },
        {
            "id": "t6",
            "content": "店名：桥头砂锅\n评分：3星\n时间：2024-03-02\n评价：汤底浓郁口感不错，但服务效率不高，上菜有时会等很久。价格属于中等偏上，偶尔有折扣活动能接受。餐厅在小巷里，门牌不够明显，第一次来的顾客可能会找不到。总体感觉中等。标签：味道好, 位置偏僻"
        }
    ]
    return docs


# -------------------------
# 测试运行示例（如何传入 qwen_apikey 的示例）
# -------------------------
if __name__ == "__main__":
    # qwen_apikey 示例（请替换为真实 key）
    example_qwen_apikey = 'sk-123456'

    # 构造抽取器，构造时会在 __init__ 中尝试构建 self.model
    extractor = EnhancedOpinionExtractorV7(qwen_apikey=example_qwen_apikey,model_id = 'qwen-plus')

    # 生成测试文档
    docs = build_test_documents()

    # 执行抽取：use_qwen_model=True 时会尝试使用 self.model（如果 self.model 存在）
    results = extractor.extract_triples(docs, use_qwen_model=True)

    # 打印结果
    for r in results:
        print(f"\n文档 {r['id']} （used_model={r['used_model']}）抽取到 {len(r['triples'])} 条观点：")
        for t in r['triples']:
            print(" ", json.dumps(t, ensure_ascii=False))

    '''
    单条文本_call_langextract抽取
    '''
    # qwen_apikey 示例（请替换为真实 key）
    example_qwen_apikey = 'sk-112223333'
    
    # 构造抽取器，构造时会在 __init__ 中尝试构建 self.model
    extractor = EnhancedOpinionExtractorV7(qwen_apikey=example_qwen_apikey,model_id = 'qwen-plus')
    
    content = '店名：老王烧烤\n评分：5星\n时间：2024-08-12\n评价：我们一家四口晚上去吃，羊肉串多汁非常好吃，老板还推荐了几款特色蘸料，味道很到位。服务员态度热情，上菜也很快。餐厅门口就有停车位，离地铁站步行约5分钟，非常方便。总体很满意，会再来。标签：味道棒, 服务好'
    
    result = extractor._call_langextract(doc["content"], model=extractor.model, extraction_passes=2)

