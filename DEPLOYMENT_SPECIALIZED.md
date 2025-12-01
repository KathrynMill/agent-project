# 专用模型部署指南

## 📋 概述

本指南详细说明如何部署基于专用AI模型的剧本压缩系统。

## 🏗️ 系统架构

### 核心组件
- **专用压缩模型**: 基于T5的深度学习模型
- **API服务**: FastAPI REST API
- **数据存储**: NebulaGraph + Qdrant
- **缓存服务**: Redis
- **监控系统**: TensorBoard + Prometheus

### 服务依赖关系
```
┌─────────────────┐    ┌─────────────────┐
│   API Service   │◄──►│  Specialized     │
│   (FastAPI)      │    │  Compression    │
└─────────┬───────┘    │     Model        │
          │           └─────────────────┘
          ▼
┌─────────────────┐
│   Data Layer    │
│ Nebula+Qdrant   │
└─────────────────┘
```

## 🔧 环境要求

### 硬件要求

#### 最低配置
- **CPU**: 4核心
- **内存**: 8GB RAM
- **存储**: 50GB 可用空间
- **网络**: 100Mbps带宽

#### 推荐配置
- **CPU**: 8核心或更多
- **内存**: 16GB RAM或更多
- **GPU**: NVIDIA GPU (推荐RTX 3080+)
- **存储**: 100GB SSD
- **网络**: 1Gbps带宽

#### 生产环境
- **CPU**: 16核心
- **内存**: 32GB RAM
- **GPU**: NVIDIA A100/V100
- **存储**: 500GB NVMe SSD
- **网络**: 10Gbps

### 软件要求

#### 基础环境
```bash
# 操作系统
Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

# Python
Python 3.8+ (推荐3.10+)

# Docker
Docker 20.10+
Docker Compose 2.0+

# NVIDIA相关 (GPU版本)
NVIDIA Driver 470+
CUDA 11.0+
```

#### 依赖包
```bash
# 系统依赖
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    htop \
    nvtop

# GPU支持 (可选)
sudo apt install -y nvidia-driver-470 nvidia-cuda-toolkit
```

## 🚀 快速部署

### 1. 克隆项目
```bash
git clone https://github.com/your-org/script-compression-system.git
cd script-compression-system
```

### 2. 环境配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
vim .env

# 关键配置项
MODEL_PATH=./models/specialized_compression_model.pth
API_HOST=0.0.0.0
API_PORT=9000
```

### 3. 模型准备

#### 方案A: 使用预训练模型
```bash
# 创建模型目录
mkdir -p models

# 下载预训练模型
wget https://releases.example.com/specialized_compression_model_v2.1.pth \
     -O models/specialized_compression_model.pth

# 验证模型完整性
sha256sum models/specialized_compression_model.pth
```

#### 方案B: 自己训练模型
```bash
# 安装训练依赖
pip install -r requirements/prod.txt

# 准备训练数据
mkdir -p data/samples
python scripts/prepare_training_data.py

# 开始训练
python scripts/train_model.py --config scripts/configs/training_config.yaml

# 评估模型
python scripts/evaluate_model.py --model-path ./checkpoints/best_model.pth
```

### 4. Docker部署

#### 使用专用配置
```bash
# 启动所有服务
docker-compose -f docker-compose.specialized.yml up -d

# 查看服务状态
docker-compose -f docker-compose.specialized.yml ps

# 查看日志
docker-compose -f docker-compose.specialized.yml logs -f api
```

#### 生产环境配置
```bash
# 设置GPU资源限制
docker-compose -f docker-compose.specialized.yml \
  --profile production up -d

# 使用外部配置文件
docker-compose -f docker-compose.specialized.yml \
  -f docker-compose.prod.yml up -d
```

### 5. 验证部署
```bash
# 健康检查
curl http://localhost:9000/api/v1/health

# 模型信息
curl http://localhost:9000/api/v1/compression/model/info

# 快速测试
python scripts/quick_test.py
```

## 🔒 安全配置

### 1. 网络安全
```yaml
# docker-compose.prod.yml
services:
  api:
    networks:
      - internal
    ports:
      - "127.0.0.1:9000:9000"  # 仅本地访问

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    networks:
      - internal
      - external
```

### 2. API安全
```bash
# 配置API密钥
echo "API_KEY=your-secure-api-key-here" >> .env

# 配置CORS
echo "CORS_ORIGINS=[\"https://yourdomain.com\"]" >> .env

# 配置SSL
echo "SSL_CERT_PATH=/etc/nginx/ssl/cert.pem" >> .env
echo "SSL_KEY_PATH=/etc/nginx/ssl/key.pem" >> .env
```

### 3. 访问控制
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location /api/ {
        proxy_pass http://api:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # API密钥验证
        auth_request /auth;
    }
}
```

## 📊 监控和日志

### 1. 应用监控
```bash
# 启动监控服务
docker-compose -f docker-compose.specialized.yml \
  --profile monitoring up -d

# TensorBoard
http://localhost:6006

# Prometheus
http://localhost:9090

# Grafana
http://localhost:3000
```

### 2. 日志配置
```python
# logging.conf
[loggers]
keys=root,uvicorn,compression

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=INFO
handlers=console,file

[logger_compression]
level=DEBUG
handlers=console,file
qualname=compression
propagate=0
```

### 3. 性能监控
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'compression-api'
    static_configs:
      - targets: ['api:9000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## 🔧 性能优化

### 1. 模型优化

#### 模型量化
```bash
# 安装量化工具
pip install optimum

# 量化模型
python scripts/quantize_model.py \
  --input-model models/specialized_compression_model.pth \
  --output-model models/quantized_model.pth
```

#### 模型蒸馏
```bash
# 训练轻量级模型
python scripts/distill_model.py \
  --teacher-model models/large_model.pth \
  --student-config configs/small_model.yaml
```

### 2. 推理优化

#### GPU优化
```python
# config.py
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
CUDA_VISIBLE_DEVICES=0
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### 批处理优化
```bash
# 环境变量
MODEL_MAX_BATCH_SIZE=4
MODEL_CACHE_SIZE=100
MODEL_CACHE_TTL=3600
```

### 3. 缓存策略
```bash
# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0

# 缓存设置
COMPRESSION_CACHE_TTL=3600
COMPRESSION_CACHE_SIZE=1000
```

## 🌐 负载均衡

### 1. Nginx配置
```nginx
upstream compression_api {
    least_conn;
    server api1:9000 weight=3 max_fails=3 fail_timeout=30s;
    server api2:9000 weight=3 max_fails=3 fail_timeout=30s;
    server api3:9000 weight=2 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://compression_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 2. Kubernetes部署
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: compression-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: compression-api
  template:
    metadata:
      labels:
        app: compression-api
    spec:
      containers:
      - name: compression-api
        image: script-compression:latest
        ports:
        - containerPort: 9000
        env:
        - name: MODEL_PATH
          value: "/app/models/specialized_compression_model.pth"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
```

## 📋 部署检查清单

### 部署前检查
- [ ] 硬件资源满足要求
- [ ] 网络配置正确
- [ ] 安全证书准备完成
- [ ] 监控系统配置完成
- [ ] 备份策略制定完成

### 部署步骤检查
- [ ] 代码下载完成
- [ ] 环境变量配置完成
- [ ] 模型文件准备完成
- [ ] Docker镜像构建完成
- [ ] 服务启动成功
- [ ] 健康检查通过

### 部署后检查
- [ ] API服务响应正常
- [ ] 模型加载成功
- [ ] 数据库连接正常
- [ ] 监控指标正常
- [ ] 日志输出正常
- [ ] 性能测试通过

## 🆘 故障排除

### 常见部署问题

#### 1. 模型加载失败
```bash
# 检查模型文件
ls -la models/
file models/specialized_compression_model.pth

# 检查权限
chmod 644 models/specialized_compression_model.pth

# 检查环境变量
echo $MODEL_PATH
```

#### 2. GPU内存不足
```bash
# 检查GPU状态
nvidia-smi
nvtop

# 调整批次大小
export MODEL_MAX_BATCH_SIZE=2

# 使用CPU推理
export CUDA_VISIBLE_DEVICES=""
```

#### 3. 端口冲突
```bash
# 检查端口占用
netstat -tulpn | grep 9000

# 修改端口
echo "API_PORT=9001" >> .env

# 使用Docker端口映射
docker run -p 9001:9000 compression-api
```

#### 4. 服务启动失败
```bash
# 查看详细日志
docker-compose -f docker-compose.specialized.yml logs api

# 进入容器调试
docker-compose -f docker-compose.specialized.yml exec api bash

# 检查配置文件
docker-compose -f docker-compose.specialized.yml config
```

### 性能问题诊断

#### 1. 响应时间过长
```bash
# 检查GPU使用率
nvidia-smi -l 1

# 检查系统资源
htop
iotop

# 分析API性能
curl -w "@curl-format.txt" http://localhost:9000/api/v1/health
```

#### 2. 内存泄漏
```bash
# 监控内存使用
watch -n 1 'free -h && docker stats'

# Python内存分析
pip install memory-profiler
python -m memory_profiler api/app.py
```

## 📞 支持和维护

### 日常维护
1. **日志清理**: 定期清理旧的日志文件
2. **模型更新**: 定期更新模型版本
3. **监控检查**: 检查监控告警和系统健康状态
4. **备份验证**: 验证数据备份的完整性

### 升级流程
1. **备份数据**: 备份模型和数据
2. **测试验证**: 在测试环境验证新版本
3. **灰度发布**: 逐步替换生产环境实例
4. **监控观察**: 监控新版本性能表现
5. **完成切换**: 全部切换到新版本

### 技术支持
- **文档**: 查看项目文档
- **社区**: GitHub Issues和讨论
- **监控**: 内部监控系统
- **告警**: 设置适当的告警通知

---

**部署成功后，您将拥有一个高性能、高可用的专用模型压缩系统！** 🎉