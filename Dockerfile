FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]