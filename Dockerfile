FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/nanotrain

COPY pyproject.toml README.md ./
COPY nanotrain ./nanotrain
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["pytest"]

