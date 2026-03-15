FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel

RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install --no-cache-dir -U openmim
RUN mim install --no-cache-dir "mmengine>=0.6.0"
RUN mim install --no-cache-dir "mmcv>=2.0.0rc4,<2.1.0"
RUN mim install --no-cache-dir "mmaction2==1.1.0"

#forcing downgrade
RUN pip install --no-cache-dir "numpy<2.0.0" pandas

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]