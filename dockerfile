# 1. Base image with Python & Poetry
FROM python:3.10-slim AS builder

# 2. Install system deps & poetry
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl build-essential \
  && curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - \
  && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# 3. Set working dir, copy project metadata
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# 4. Install Python deps (no dev)
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# 5. Copy your source
COPY . /app

# 6. (Optional) expose ports if you add a web server later
# EXPOSE 8000

# 7. Default command 
CMD ["poetry", "run", "python", "-m", "src.train_model"]
