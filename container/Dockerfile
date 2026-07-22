FROM python:3.12.12-slim-trixie@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOME=/home/app \
    PORT=8080

RUN apt-get update \
    && apt-get install --yes --no-install-recommends espeak-ng=1.52.0+dfsg-5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt

COPY bakeoff.yaml ./bakeoff.yaml
COPY rules ./rules
COPY src ./src
RUN espeak-ng --version | grep '1.52.0' \
    && python -c 'from earshift_bakeoff.gates import build_gate_database; build_gate_database()' \
    && python -c 'from earshift_bakeoff.deploy_service import DeployService; assert DeployService().health()["status"] == "ok"' \
    && useradd --system --uid 10001 --create-home --home-dir /home/app app

EXPOSE 8080
USER 10001:10001
CMD ["python", "-m", "earshift_bakeoff.deploy_service"]
