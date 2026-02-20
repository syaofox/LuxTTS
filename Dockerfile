# LuxTTS CPU 版 Docker 镜像
# 多阶段构建 + 非 root 用户 + 体积优化

# ========== 阶段 1: 构建依赖 ==========
FROM python:3.11-slim-bookworm AS builder

# 构建期系统依赖（编译部分 Python 包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先安装 PyTorch CPU 版（避免拉取 CUDA 版本，节省约 2–3GB）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖（使用 requirements-docker.txt 避免重复安装 CUDA 版 torch）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ========== 阶段 2: 运行时镜像 ==========
FROM python:3.11-slim-bookworm AS runtime

# 运行时系统依赖（soundfile/librosa 需要 libsndfile）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非 root 用户
RUN groupadd -r luxtts --gid=1000 \
    && useradd -r -g luxtts --uid=1000 --create-home --shell=/bin/bash luxtts

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY --chown=luxtts:luxtts zipvoice/ ./zipvoice/
COPY --chown=luxtts:luxtts api_server.py app.py ./

# 参考音频目录（可挂载覆盖） ckpt目录
RUN mkdir -p ref_audio ckpt && chown -R luxtts:luxtts ref_audio ckpt /app 

USER luxtts

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LUXTTS_REF_AUDIO=""

EXPOSE 8765

# 默认启动 API 服务
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1", "--app-dir", "/app"]
