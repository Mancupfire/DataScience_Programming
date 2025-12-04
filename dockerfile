FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng Docker cache layer
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở port (8501 là port mặc định của Streamlit, 8888 nếu bạn muốn chạy Jupyter)
EXPOSE 8501
EXPOSE 8888

# Lệnh chạy mặc định khi container khởi động
# Giả định app.py là Streamlit app. Nếu là file python thường, đổi thành ["python", "app.py"]
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
