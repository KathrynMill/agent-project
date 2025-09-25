# 部署指南

本文档介绍如何部署剧本杀 Agent 系统到生产环境。

## 生产环境部署

### 1. 服务器要求

#### 最低配置
- CPU: 4 核心
- 内存: 16GB RAM
- 存储: 100GB SSD
- 网络: 100Mbps

#### 推荐配置
- CPU: 8 核心
- 内存: 32GB RAM
- 存储: 500GB SSD
- GPU: NVIDIA RTX 4090 或更高（可选）
- 网络: 1Gbps

### 2. 系统要求
- Ubuntu 20.04+ 或 CentOS 8+
- Docker 24+
- Docker Compose v2
- NVIDIA Container Toolkit（如果使用 GPU）

### 3. 环境配置

#### 创建生产环境配置
```bash
# 复制环境配置模板
cp .env.example .env.production

# 编辑生产环境配置
nano .env.production
```

#### 生产环境变量
```bash
# API 配置
OPENAI_API_BASE=http://vllm:8000/v1
OPENAI_API_KEY=your-production-key
EMBEDDINGS_API=http://embeddings:80

# 数据库配置
QDRANT_URL=http://qdrant:6333
NEBULA_ADDRS=nebula-graphd0:9669,nebula-graphd1:9669
NEBULA_USER=root
NEBULA_PASSWORD=your-secure-password
NEBULA_SPACE=scripts

# 安全配置
JWT_SECRET_KEY=your-jwt-secret
CORS_ORIGINS=https://yourdomain.com
```

### 4. 安全配置

#### 启用 NebulaGraph 认证
```yaml
# 在 docker-compose.yml 中修改
nebula-graphd0:
  environment:
    - --enable_authorize=true
    - --auth_type=password
```

#### 配置防火墙
```bash
# 只开放必要端口
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 9000  # API (如果直接暴露)
sudo ufw enable
```

#### 使用反向代理
```nginx
# nginx 配置示例
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 5. 部署步骤

#### 使用 Docker Compose
```bash
# 克隆项目
git clone https://github.com/KathrynMill/agent-project.git
cd agent-project

# 配置环境变量
cp .env.example .env
nano .env

# 启动服务
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 检查服务状态
docker compose ps

# 查看日志
docker compose logs -f
```

#### 使用 Kubernetes
```bash
# 创建命名空间
kubectl create namespace script-agent

# 部署配置
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/ingress.yaml
```

### 6. 监控和日志

#### 配置 Prometheus 监控
```yaml
# docker-compose.monitoring.yml
version: "3.9"
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

#### 配置日志收集
```yaml
# 在 docker-compose.yml 中添加
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 7. 备份策略

#### 数据库备份
```bash
# NebulaGraph 备份
docker exec nebula-storaged0 /usr/local/nebula/bin/nebula-backup --meta_server_addrs=nebula-metad0:9559 --backup_name=backup_$(date +%Y%m%d_%H%M%S)

# Qdrant 备份
docker exec qdrant /qdrant/qdrant --config-path /qdrant/config/production.yaml backup --output-dir /backup/$(date +%Y%m%d_%H%M%S)
```

#### 自动备份脚本
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/$DATE"

mkdir -p $BACKUP_DIR

# 备份 NebulaGraph
docker exec nebula-storaged0 /usr/local/nebula/bin/nebula-backup --meta_server_addrs=nebula-metad0:9559 --backup_name=backup_$DATE

# 备份 Qdrant
docker exec qdrant /qdrant/qdrant --config-path /qdrant/config/production.yaml backup --output-dir $BACKUP_DIR/qdrant

# 压缩备份
tar -czf "/backup/backup_$DATE.tar.gz" $BACKUP_DIR

# 清理旧备份（保留7天）
find /backup -name "backup_*.tar.gz" -mtime +7 -delete
```

### 8. 性能优化

#### 调整 Docker 资源限制
```yaml
# docker-compose.prod.yml
services:
  vllm:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'

  nebula-graphd0:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

#### 配置缓存
```python
# 在 API 中添加 Redis 缓存
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### 9. 故障排除

#### 常见问题
1. **服务启动失败**
   ```bash
   # 检查日志
   docker compose logs service-name
   
   # 检查资源使用
   docker stats
   ```

2. **内存不足**
   ```bash
   # 增加交换空间
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **网络连接问题**
   ```bash
   # 检查网络配置
   docker network ls
   docker network inspect agent-project_default
   ```

### 10. 更新和维护

#### 滚动更新
```bash
# 更新单个服务
docker compose pull service-name
docker compose up -d service-name

# 更新所有服务
docker compose pull
docker compose up -d
```

#### 健康检查
```bash
# 检查服务健康状态
curl http://localhost:9000/health
curl http://localhost:9000/nebula/ping
curl http://localhost:9000/vector/info
```

## 云部署

### AWS 部署
- 使用 ECS 或 EKS
- 配置 RDS 和 ElastiCache
- 使用 CloudFront 做 CDN

### Azure 部署
- 使用 AKS
- 配置 Azure Database
- 使用 Azure CDN

### GCP 部署
- 使用 GKE
- 配置 Cloud SQL
- 使用 Cloud CDN

## 成本优化

### 资源优化
- 使用 Spot 实例
- 配置自动扩缩容
- 优化模型大小

### 存储优化
- 使用对象存储
- 配置生命周期策略
- 压缩备份文件

## 安全最佳实践

1. 定期更新依赖
2. 使用 HTTPS
3. 配置访问控制
4. 监控异常活动
5. 定期安全审计
