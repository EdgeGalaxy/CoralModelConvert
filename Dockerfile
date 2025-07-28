# 第一阶段：构建依赖环境
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.5 python3 -

# 将Poetry添加到PATH
ENV PATH="/root/.local/bin:$PATH"

# 先复制依赖文件以利用Docker缓存
COPY pyproject.toml poetry.lock ./

# 配置Poetry并安装依赖到虚拟环境
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root --only=main

# 第二阶段：运行时环境
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制项目文件
COPY . .

# 设置PATH使用虚拟环境
ENV PATH="/app/.venv/bin:$PATH"

# 创建必要的目录
RUN mkdir -p temp output logs

# 环境变量
ENV HOST=0.0.0.0 \
    PORT=8000

# 暴露端口
EXPOSE 8000

CMD ["python", "run.py"]