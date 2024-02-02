FROM python:3.11

WORKDIR /code

# Set your GitHub repo URL here
ARG GIT_REPO_URL=https://github.com/xenophonlabs/crvUSDrisk.git
ARG GIT_BRANCH=dev  # Adjust the branch name as necessary

# Install system dependencies required for gmpy2 compilation
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* 

RUN git lfs install
RUN git clone --branch $GIT_BRANCH $GIT_REPO_URL . && git lfs pull

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/code"

CMD ["python", "./app/main.py"]
