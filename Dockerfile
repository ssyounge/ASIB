FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install .[dev]
CMD ["python", "-m", "tests.smoke"]
