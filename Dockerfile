FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Install build-essential karena Prophet butuh compiler C++
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .

# Upgrade pip dan install semua library
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh file project (termasuk folder export_model)
COPY . .

# Buka port 8000
EXPOSE 8000

# Jalankan uvicorn (Perhatikan: kita pakai main:app karena nama file-nya main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
