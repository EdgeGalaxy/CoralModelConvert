# Coral Model Convert Service

一个基于FastAPI的统一模型转换服务，支持ONNX到RKNN格式的转换。

## 功能特性

- **RKNN转换**: 将ONNX模型转换为RKNN格式，支持RK平台
- **统一接口**: 可扩展的适配器模式，便于支持更多模型格式
- **异步处理**: 后台任务处理，支持大模型转换
- **RESTful API**: 清晰的REST API接口，带完整文档
- **错误处理**: 健壮的错误处理和参数验证

## 支持的转换

- ONNX → RKNN (RK3562, RK3566, RK3568, RK3588)

## 快速开始

### 安装依赖

```bash
# 使用Poetry安装
poetry install

# 或使用pip
pip install -r requirements.txt
```

### 运行服务

```bash
# 开发模式
python run.py

# 或使用uvicorn
uvicorn coral_model_convert.main:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 `http://localhost:8000` 启动

### API文档

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker部署

### 使用Docker Compose

```bash
# 生产环境
docker-compose up -d

# 开发环境
docker-compose -f docker-compose.dev.yml up
```

### 直接使用Docker

```bash
# 构建镜像
docker build -t coral-model-convert .

# 运行容器
docker run -d \
  --name coral-model-convert \
  -p 8000:8000 \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/output:/app/output \
  coral-model-convert
```

## API使用

### 转换ONNX到RKNN

```bash
curl -X POST "http://localhost:8000/api/v1/convert/rknn" \
  -H "Content-Type: multipart/form-data" \
  -F "model_file=@model.onnx" \
  -F "target_platform=rk3588" \
  -F "hybrid_quant=true"
```

### 检查任务状态

```bash
curl "http://localhost:8000/api/v1/tasks/{task_id}"
```

### 下载结果

```bash
curl "http://localhost:8000/api/v1/tasks/{task_id}/download" -o model.rknn
```

## 配置

主要配置选项在 `coral_model_convert/config.py` 中：

- `MAX_FILE_SIZE`: 最大上传文件大小（默认: 500MB）
- `ALLOWED_EXTENSIONS`: 支持的文件格式
- `RKNN_SUPPORTED_PLATFORMS`: 支持的RKNN平台
- `TEMP_DIR`, `OUTPUT_DIR`: 文件存储目录

## 项目结构

```
coral_model_convert/
├── __init__.py
├── main.py              # FastAPI应用主文件
├── config.py            # 配置设置
├── adapter.py           # 统一转换器接口
├── models.py            # Pydantic模型
├── tasks.py             # 任务管理
├── exceptions.py        # 自定义异常
├── error_handlers.py    # 全局错误处理
├── api/
│   ├── __init__.py
│   └── conversion.py    # API接口
└── converters/
    ├── __init__.py
    └── rknn_converter.py # RKNN转换逻辑
```

## 开发

### 添加新的转换器

要添加对新模型格式的支持：

1. 创建实现 `BaseModelConverter` 的转换器类
2. 在 `ModelConverterAdapter._register_converters()` 中注册转换器
3. 根据需要添加相应的API接口

## GitHub Actions

项目包含自动化CI/CD流水线：

- **build.yml**: 构建和推送Docker镜像

### 必需的Secrets

在GitHub仓库中配置以下secrets：

```
DOCKER_USERNAME: Docker Hub用户名
DOCKER_PASSWORD: Docker Hub访问令牌
DOCKERHUB_REGISTRY: 可选的自定义注册表URL
```