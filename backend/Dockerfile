FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci --omit=dev

COPY gait-analysis/requirements.txt ./gait-analysis/requirements.txt
COPY therapy-exercises/requirements.txt ./therapy-exercises/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r ./gait-analysis/requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages -r ./therapy-exercises/requirements.txt

COPY patch-frozendict.sh /usr/local/bin/patch-frozendict.sh
RUN chmod +x /usr/local/bin/patch-frozendict.sh && /usr/local/bin/patch-frozendict.sh

COPY . .

ENV NODE_ENV=production
ENV PORT=8080
ENV GAIT_ANALYSIS_PORT=5001
ENV THERAPY_PORT=5002
ENV USE_GLOBAL_PYTHON=true
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["node", "start.js"]
