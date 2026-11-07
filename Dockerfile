FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-liberation fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY .streamlit/ .streamlit/
COPY code/ code/
COPY data/ data/
COPY ketqua/ ketqua/
COPY baocao/*.pdf baocao/*.docx baocao/

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "code/app.py"]
