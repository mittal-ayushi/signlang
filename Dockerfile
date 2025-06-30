# use python 3.10 so tensorflow works
FROM python:3.10-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy requirements first and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of your app
COPY . .

# run your app
CMD ["gunicorn", "app:app"]

