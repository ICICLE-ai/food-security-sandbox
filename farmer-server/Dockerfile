FROM python:3.13-slim-bookworm
LABEL author="icicle-ai"
RUN useradd -ms /bin/bash icicle
RUN mkdir /app && chown icicle /app

USER icicle
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./uv.lock .
COPY ./pyproject.toml .

RUN uv sync --frozen --no-cache
RUN rm uv.lock pyproject.toml

RUN mkdir list
COPY ./pca_model.pkl .
COPY ./app.py .
COPY ./config.py .

CMD ["uv", "run", "app.py"]
