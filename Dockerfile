# 第一阶段：构建依赖环境
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖并清理缓存
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g; s/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# 安装Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.5 python3 - \
    && rm -rf /tmp/*

# 将Poetry添加到PATH
ENV PATH="/root/.local/bin:$PATH"

# 先复制依赖文件以利用Docker缓存
COPY pyproject.toml poetry.lock ./

# 配置Poetry并安装依赖到虚拟环境，同时清理缓存
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root --only=main && \
    poetry cache clear --all pypi && \
    rm -rf /tmp/*

# 第二阶段：运行时环境
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装运行时依赖并清理缓存
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g; s/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# 从构建阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制项目文件
COPY . .

RUN chmod +x /app/bootstrap

# 设置PATH使用虚拟环境
ENV PATH="/app/.venv/bin:$PATH"

# 创建必要的目录
RUN mkdir -p temp output logs

# 环境变量
ENV HOST=0.0.0.0 \
    PORT=8000

# 暴露端口
EXPOSE 8000

CMD ["./bootstrap"]
