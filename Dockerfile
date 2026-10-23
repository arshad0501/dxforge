FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Re-train if the shipped artefacts cannot be loaded with this image's library versions.
RUN python -c "from dxforge.ml.inference import load_intent_model as l; import sys; sys.exit(0 if l()[0] is not None else 1)" \
    || python scripts/train.py --quick
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
