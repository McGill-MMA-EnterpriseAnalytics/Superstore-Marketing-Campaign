#########################################
# Dockerfile for Training
# This Dockerfile is for training models only
#########################################
FROM python:3.10-slim AS builder

# system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only poetry config to speed up rebuilds
COPY pyproject.toml poetry.lock* /app/

# install Poetry (latest), disable venvs, install only main deps
RUN pip install poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --without dev --no-root

#########################################
# Runtime stage: copy just what's needed
#########################################
FROM python:3.10-slim AS runtime

# install only the OpenMP runtime so lightgbm can find libgomp.so.1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry

# copy your code
COPY src/ /app/src/
COPY src/utils/ /app/utils/
COPY config.yaml /app/config.yaml

# create models directory
RUN mkdir -p /app/models/

# set PYTHONPATH so you can import your package
ENV PYTHONPATH=/app

# default command - run training script
CMD ["python", "-m", "src.train_model"]