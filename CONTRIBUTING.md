# 贡献指南

感谢您对剧本杀 Agent 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 1. Fork 项目
- 点击 GitHub 页面右上角的 "Fork" 按钮
- 将项目 fork 到您的 GitHub 账户

### 2. 克隆项目
```bash
git clone https://github.com/您的用户名/agent-project.git
cd agent-project
```

### 3. 创建分支
```bash
git checkout -b feature/您的功能名称
```

### 4. 进行开发
- 修改代码
- 添加测试
- 更新文档

### 5. 提交更改
```bash
git add .
git commit -m "添加: 描述您的更改"
```

### 6. 推送分支
```bash
git push origin feature/您的功能名称
```

### 7. 创建 Pull Request
- 在 GitHub 上创建 Pull Request
- 详细描述您的更改

## 开发环境设置

### 前置要求
- Docker 24+
- Docker Compose v2
- Python 3.11+（用于本地开发）

### 启动开发环境
```bash
# 启动所有服务
docker compose up -d --build

# 检查服务状态
docker compose ps

# 运行测试
python test_api.py
```

## 代码规范

### Python 代码
- 使用 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 使用类型提示

### 提交信息
- 使用中文描述
- 格式：`类型: 简短描述`
- 类型包括：添加、修复、更新、删除、重构等

## 报告问题

如果您发现 bug 或有功能建议，请：

1. 检查是否已有相关 issue
2. 创建新的 issue
3. 提供详细的复现步骤
4. 包含环境信息

## 功能开发

### 当前优先级功能
- [ ] 支持更多剧本格式（PDF、Word）
- [ ] 增加推理规则引擎
- [ ] 添加用户界面
- [ ] 支持多语言
- [ ] 性能优化

### 架构扩展
- [ ] 添加监控和日志
- [ ] 实现用户认证
- [ ] 支持分布式部署
- [ ] 添加缓存机制

## 测试

### 运行测试
```bash
# 运行所有测试
python test_api.py

# 测试特定功能
curl -X GET http://localhost:9000/health
```

### 添加新测试
- 在 `test_api.py` 中添加新的测试函数
- 确保测试覆盖新功能
- 测试应该能够独立运行

## 文档

### 更新文档
- 修改 `README.md` 以反映新功能
- 添加 API 文档
- 更新安装和使用说明

## 许可证

本项目使用 MIT 许可证。贡献代码即表示您同意将代码在 MIT 许可证下发布。

## 联系方式

如有问题，请通过以下方式联系：
- 创建 GitHub Issue
- 发送邮件到项目维护者

感谢您的贡献！
