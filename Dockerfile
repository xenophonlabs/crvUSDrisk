FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/code"

# Install system dependencies required for gmpy2 compilation
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app/ /code/app/
COPY ./src/ /code/src/
# COPY ./results/ /code/results/

CMD ["python", "./app/main.py"]
