FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY web_app/ ./web_app/

RUN python web_app/train_model.py

EXPOSE 5000

WORKDIR /app/web_app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]