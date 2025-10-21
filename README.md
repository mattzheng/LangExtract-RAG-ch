# 中文大众点评评论 RAG 演示（langextract_rag_cn.py）

这是基于示例 `langextract_rag.py` 改写的中文版本示例工程，场景为解析中文大众点评风格的评论并做基于 metadata 的检索演示。项目保持原示例的设计理念：优先使用 LangExtract 做结构化抽取，失败回退到正则抽取；索引层为内存级别的 SmartVectorStore（未做真正的向量化），检索基于元数据模糊匹配与文本子串匹配。

## 文件列表
- `langextract_rag_cn.py`：主脚本（中文版本），实现了：
  - 静态样本文档（get_sample_documents）
  - 优先使用 `langextract` 做元数据抽取，未安装则回退到中文正则抽取
  - 内存级 SmartVectorStore：模拟索引与检索（基于元数据匹配和子串匹配）
  - `extract_smart_filters`：从中文查询中抽取过滤条件（店名、评分、关注点、情感）
  - 演示查询流程并打印检索结果
- `requirements.txt`：运行所需的 Python 包列表
- `README.md`：此说明文件

## 环境与依赖
- Python 版本建议 3.8+
- 安装依赖：
  - 推荐在虚拟环境中安装：
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS / Linux
    .venv\Scripts\activate     # Windows (PowerShell)
    pip install -r requirements.txt
    ```
- 说明：
  - `langextract` 是可选的：若安装了 `langextract`，脚本会调用其抽取能力；若未安装，脚本会降级为内置的正则抽取逻辑（仍可运行）。
  - `python-dotenv` 用于加载环境变量（示例中保留以与原项目一致）。

## 快速运行
在安装依赖并激活虚拟环境后，运行：
```bash
python langextract_rag_cn.py
```
脚本会：
1. 加载示例评论
2. 对每条评论执行元数据抽取（langextract 或正则回退）
3. 将结果索引到内存级 SmartVectorStore
4. 对若干测试查询执行带/不带过滤器的检索并打印输出

## 输出示例（简要）
运行后你会看到类似的流程输出：
- 每条文档被处理和抽取的 metadata
- 索引成功提示
- 针对测试查询展示“使用元数据过滤检索到的结果数”与“不使用过滤检索到的结果数”，以及具体匹配文档 ID 与 metadata。

## 设计限制与注意事项
该演示遵循原示例的简化设计，因此存在以下限制（在 README 中明确提示，便于快速理解和评估）：
- 未做文本分块（chunking）：metadata 与文本为文档级别，长文本细粒度信息会被合并。
- 未做向量化或语义检索：SmartVectorStore 仅基于字符串子串匹配与简单的元数据模糊匹配，无法处理复杂的语义同义替换或隐含表达。
- 正则抽取策略较为脆弱：真实用户评论表达多样，正则会漏抽或误抽。项目中保留了作为 fallback 的正则，但建议在生产场景使用 NER/模型增强抽取或引入 LangExtract。
- 索引规模和性能：当前索引为内存线性扫描，不适合大规模数据（万级以上）。生产场景应接入向量数据库或倒排索引并实现持久化。
- 情感/评分判定为简单规则，存在否定/复合句误判风险，建议使用专门的情感分类模型做稳健判断。

## 推荐后续改进（非必须）
- 对评论做 chunking 并为每个 chunk 提取 chunk 级 metadata。
- 引入 embedding + 向量数据库（FAISS / Chroma / Pinecone / Weaviate）以实现语义检索。
- 将 `extract_smart_filters` 的规则替换/增强为基于中文 NER/意图识别的小模型。
- 增加抽取置信度、审计信息、异常重试、日志与监控。
- 做去重（based on fingerprint 或 embedding 相似度）与聚合统计。

## 如何贡献与测试
- 若你修改或扩展了抽取逻辑，建议在脚本中加入单元测试与标注样本来验证抽取准确率。
- 若要将内存索引替换为向量 DB，可先在小数据集上做 proof-of-concept，再迁移到生产环境。

## 许可与免责声明
该示例代码用于演示与学习目的。若用于生产，请注意：
- 数据隐私与合规（用户评论可能包含 PII），在采集/索引前做好脱敏与访问控制。
- 若使用外部 LLM/embedding 服务，留意成本与速率限制。
