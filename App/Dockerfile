FROM python:3.13-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Nur requirements kopieren → schnelleres Caching
COPY requirements.txt .

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# Projektdateien kopieren
COPY pipe.py .
COPY helpers/ helpers/
COPY data/ data/

# Port für Dash
EXPOSE 8050

# Startkommando
CMD ["python", "pipe.py"]
