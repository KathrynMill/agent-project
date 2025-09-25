# 项目结构说明

本文档详细说明剧本杀 Agent 项目的文件结构和各个组件的作用。

## 项目根目录

```
agent-project/
├── .github/                    # GitHub 配置
│   └── workflows/              # CI/CD 工作流
│       └── ci.yml             # 持续集成配置
├── api/                       # 后端 API 服务
│   ├── app/                   # 应用代码
│   │   ├── main.py           # FastAPI 主应用
│   │   ├── nebula_init.nGQL  # NebulaGraph 初始化脚本
│   │   └── services/         # 服务层
│   │       ├── llm_service.py      # LLM 服务
│   │       ├── nebula_service.py   # NebulaGraph 服务
│   │       └── vector_service.py   # 向量数据库服务
│   ├── Dockerfile            # API 容器配置
│   └── requirements.txt      # Python 依赖
├── .dockerignore             # Docker 忽略文件
├── .gitignore               # Git 忽略文件
├── CONTRIBUTING.md          # 贡献指南
├── DEPLOYMENT.md            # 部署指南
├── docker-compose.yml       # Docker 编排配置
├── LICENSE                  # MIT 许可证
├── PROJECT_STRUCTURE.md     # 项目结构说明（本文件）
├── README.md               # 项目说明
├── sample_script.txt       # 样例剧本
└── test_api.py            # API 测试脚本
```

## 详细说明

### 1. 后端 API 服务 (`api/`)

#### 主应用 (`api/app/main.py`)
- FastAPI 应用入口
- 定义所有 API 端点
- 处理 HTTP 请求和响应
- 集成各个服务模块

**主要端点：**
- `GET /health` - 健康检查
- `GET /nebula/ping` - NebulaGraph 连接测试
- `POST /nebula/query` - 直接执行 nGQL 查询
- `POST /text-to-ngql` - 自然语言转 nGQL
- `POST /extract` - 剧本文本抽取实体/事件并入库
- `POST /rag/query` - 完整 RAG 问答
- `GET /vector/info` - 向量库信息
- `POST /vector/search` - 向量相似性搜索

#### 服务层 (`api/app/services/`)

**LLM 服务 (`llm_service.py`)**
- 封装 LLM API 调用
- 实现文本生成功能
- 实体和事件抽取
- Text-to-nGQL 转换
- 基于检索结果的答案生成

**NebulaGraph 服务 (`nebula_service.py`)**
- 管理 NebulaGraph 连接池
- 执行 nGQL 查询
- 实体和关系的增删改查
- 批量数据导入
- 图谱搜索功能

**向量服务 (`vector_service.py`)**
- 管理 Qdrant 向量数据库
- 文本嵌入生成
- 向量相似性搜索
- 文档存储和检索
- 批量文档处理

#### 数据库初始化 (`api/app/nebula_init.nGQL`)
- 创建 NebulaGraph 空间
- 定义节点标签（Person, Location, Item, Event, Timeline）
- 定义关系类型（LOCATED_IN, HAS_ITEM, PARTICIPATED_IN, HAPPENED_AT, HAPPENED_ON）
- 创建索引优化查询性能

### 2. 容器化配置

#### Docker Compose (`docker-compose.yml`)
定义完整的服务栈：

**核心服务：**
- `vllm` - LLM 推理服务（Qwen2.5-14B-Instruct）
- `embeddings` - 文本嵌入服务（BGE-M3）
- `qdrant` - 向量数据库
- `api` - FastAPI 后端服务

**NebulaGraph 分布式集群：**
- `nebula-metad0/1/2` - 元数据服务（3个节点）
- `nebula-graphd0/1` - 图计算服务（2个节点）
- `nebula-storaged0/1/2` - 存储服务（3个节点）

#### API Dockerfile (`api/Dockerfile`)
- 基于 Python 3.11-slim
- 安装项目依赖
- 配置运行环境
- 暴露 9000 端口

### 3. 测试和文档

#### 测试脚本 (`test_api.py`)
- 健康检查测试
- NebulaGraph 连接测试
- 实体抽取测试
- RAG 问答测试
- 向量搜索测试

#### 样例数据 (`sample_script.txt`)
- 完整的剧本杀案例
- 包含人物、地点、事件、线索
- 用于测试和演示

### 4. 配置和部署

#### 环境配置
- `.env.example` - 环境变量模板
- `.gitignore` - Git 忽略规则
- `.dockerignore` - Docker 构建忽略规则

#### CI/CD 配置 (`.github/workflows/ci.yml`)
- 代码质量检查（flake8）
- 安全扫描（bandit, trivy）
- 单元测试和集成测试
- Docker 镜像构建
- 自动部署

### 5. 文档

#### 项目文档
- `README.md` - 项目概述和快速开始
- `CONTRIBUTING.md` - 贡献指南
- `DEPLOYMENT.md` - 生产环境部署指南
- `PROJECT_STRUCTURE.md` - 项目结构说明（本文件）

#### 许可证
- `LICENSE` - MIT 开源许可证

## 数据流

### 1. 剧本处理流程
```
剧本文本 → LLM抽取 → JSON数据 → NebulaGraph入库 → 向量化存储
```

### 2. 问答流程
```
用户问题 → Text-to-nGQL → 图谱查询 → 向量检索 → LLM生成答案
```

### 3. 服务交互
```
API Gateway → LLM Service → NebulaGraph Service → Vector Service
```

## 扩展点

### 1. 新增实体类型
- 在 `nebula_init.nGQL` 中添加新的 TAG
- 在 `llm_service.py` 中更新抽取规则
- 在 `nebula_service.py` 中添加对应的 upsert 方法

### 2. 新增关系类型
- 在 `nebula_init.nGQL` 中添加新的 EDGE
- 在 `llm_service.py` 中更新关系抽取逻辑
- 在 `nebula_service.py` 中添加关系创建方法

### 3. 新增 API 端点
- 在 `main.py` 中添加新的路由
- 在相应的服务中添加业务逻辑
- 更新测试脚本

### 4. 集成新的 LLM
- 在 `llm_service.py` 中添加新的模型支持
- 更新 Docker Compose 配置
- 调整环境变量

## 性能优化

### 1. 数据库优化
- 添加更多索引
- 优化查询语句
- 配置连接池

### 2. 缓存策略
- 添加 Redis 缓存
- 实现查询结果缓存
- 配置缓存过期策略

### 3. 并发处理
- 使用异步处理
- 配置工作进程
- 实现负载均衡

## 监控和日志

### 1. 应用监控
- 添加 Prometheus 指标
- 配置 Grafana 仪表板
- 实现健康检查

### 2. 日志管理
- 结构化日志输出
- 日志轮转配置
- 集中日志收集

### 3. 错误处理
- 全局异常处理
- 错误码标准化
- 错误追踪和报告

## 安全考虑

### 1. 认证授权
- JWT Token 认证
- 角色权限控制
- API 访问限制

### 2. 数据安全
- 敏感数据加密
- 传输层安全
- 数据备份策略

### 3. 网络安全
- 防火墙配置
- 反向代理设置
- DDoS 防护

这个项目结构设计遵循了微服务架构的最佳实践，具有良好的可扩展性和可维护性。
