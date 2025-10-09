FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#COPY helpers ./helpers
COPY app.py app.py

#COPY . .

RUN mkdir -p /app/output

EXPOSE 8050

CMD ["python", "app.py"]
