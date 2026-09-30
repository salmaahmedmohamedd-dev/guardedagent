FROM python:3.12-slim

# no .pyc files inside the container, and logs show up immediately in docker logs
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# layer 3: a normal user with no password and no admin rights. if the agent were ever tricked into
# running something, it runs as this user, not as root
RUN useradd --create-home --uid 10001 agent

COPY . .

USER agent

# the web page. docker-compose only publishes it on 127.0.0.1, so it is not reachable from the network
EXPOSE 8000

# docker marks the container unhealthy if the server stops answering
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["python", "server.py"]
#building an image with lightweight
#use /app working dir
#copy req.txt to my current dir
#copy the rest into /app (.dockerignore keeps .env and venv out)
#run as a non-root user and start the web server instead of the one-off agent.py demo