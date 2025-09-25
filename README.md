## 剧本杀 Agent - NebulaGraph 分布式免费方案（完整版）

本项目提供企业级、可自部署且完全免费的剧本杀智能体系统：

### 核心功能
- **NebulaGraph 分布式集群**（3 metad / 2 graphd / 3 storaged）
- **vLLM + Qwen2.5 Instruct**（OpenAI 兼容推理服务）
- **BGE-M3 嵌入服务**（HuggingFace text-embeddings-inference）
- **Qdrant 向量数据库**（语义检索）
- **完整 RAG 流程**（Text-to-nGQL + KG检索 + 向量检索 + LLM生成）

### 主要 API 端点
- `/health` - 健康检查
- `/nebula/ping` - NebulaGraph 连接测试
- `/nebula/query` - 直接执行 nGQL 查询
- `/text-to-ngql` - 自然语言转 nGQL
- `/extract` - 剧本文本抽取实体/事件并入库
- `/rag/query` - 完整 RAG 问答（KG + 向量 + LLM）
- `/vector/info` - 向量库信息
- `/vector/search` - 向量相似性搜索

### 先决条件
- Docker 24+ 与 Docker Compose v2
-（可选）NVIDIA GPU 以加速 vLLM（否则可改 CPU 模式）

### 快速开始

1. **启动所有服务**：
```bash
docker compose up -d --build
```

2. **等待服务启动**（约 2-3 分钟）：
```bash
# 检查服务状态
docker compose ps
```

3. **初始化 NebulaGraph Schema**：
```bash
# 方法1：使用 API 初始化
curl -X POST http://localhost:9000/nebula/query \
  -H 'Content-Type: application/json' \
  -d '{"nql":"CREATE SPACE IF NOT EXISTS scripts(partition_num=5, replica_factor=1, vid_type=FIXED_STRING(64)); USE scripts; CREATE TAG IF NOT EXISTS Person(name string, role string); CREATE TAG IF NOT EXISTS Location(name string); CREATE TAG IF NOT EXISTS Item(name string, type string); CREATE TAG IF NOT EXISTS Event(name string, description string); CREATE TAG IF NOT EXISTS Timeline(name string, start string, end string);"}'

# 方法2：使用测试脚本
python test_api.py
```

4. **测试完整流程**：
```bash
# 抽取样例剧本
curl -X POST http://localhost:9000/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"晚上8点，富商王建国在自家别墅举办生日晚宴。参加者包括王建国、李美丽、张律师等。"}'

# RAG 问答
curl -X POST http://localhost:9000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"王建国在哪里被发现死亡？"}'
```

### 服务端点
- **vLLM(OpenAI 兼容)**：`http://localhost:8000/v1`
- **Embeddings**：`http://localhost:8080`
- **Qdrant**：`http://localhost:6333`
- **Nebula GraphD**：`localhost:9669`
- **API**：`http://localhost:9000`

### 使用示例

#### 1. 抽取剧本并入库
```python
import requests

# 读取剧本文件
with open("sample_script.txt", "r", encoding="utf-8") as f:
    script_text = f.read()

# 抽取实体和事件
response = requests.post("http://localhost:9000/extract", 
                        json={"text": script_text})
print(response.json())
```

#### 2. 自然语言问答
```python
# 直接问答
response = requests.post("http://localhost:9000/rag/query",
                        json={"question": "谁是凶手？"})
print(response.json()["answer"])
```

#### 3. 查看生成的 nGQL
```python
# 查看问题对应的查询语句
response = requests.post("http://localhost:9000/text-to-ngql",
                        json={"question": "王建国和谁有关系？"})
print(response.json()["nql"])
```

### 架构说明

#### 知识图谱 Schema
- **节点**：Person（人物）、Location（地点）、Item（物品）、Event（事件）、Timeline（时间线）
- **关系**：LOCATED_IN、HAS_ITEM、PARTICIPATED_IN、HAPPENED_AT、HAPPENED_ON

#### RAG 流程
1. **问题理解**：自然语言 → nGQL 查询
2. **图谱检索**：执行 nGQL → 结构化事实
3. **向量检索**：语义相似性搜索 → 相关文本片段
4. **答案生成**：LLM 基于检索结果生成有根据的回答

### 注意事项
- 若无 GPU，请修改 `docker-compose.yml` 中 vLLM 配置，移除 GPU 限制或使用较小模型
- NebulaGraph 默认未开启授权，生产环境请配置安全设置
- 首次启动需要下载模型，请确保网络连接正常

### 故障排除
```bash
# 查看服务日志
docker compose logs api
docker compose logs vllm
docker compose logs nebula-graphd0

# 重启特定服务
docker compose restart api

# 完全重建
docker compose down -v
docker compose up -d --build
```



