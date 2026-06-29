FROM python:3.11-slim

WORKDIR /app

# System deps for Playwright
RUN apt-get update && apt-get install -y \
    chromium \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy app code
COPY . .

# Set Python path so src/ imports work
ENV PYTHONPATH=/app

# Streamlit port HF expects
EXPOSE 7860

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]