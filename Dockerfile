FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PILOT_DATA=/app/data MPLCONFIGDIR=/tmp/mpl
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pilot ./pilot
COPY tests ./tests

# Run as a non-root user; results are written to the mounted /app/data volume.
RUN useradd --create-home pilot && mkdir -p /app/data && chown -R pilot /app
USER pilot

ENTRYPOINT ["python", "-m", "pilot"]
CMD ["all", "--providers", "mock,mock-b", "--repeats", "3", "--run", "demo_mock"]
