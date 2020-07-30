ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} as base

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV C_FORCE_ROOT 1

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    git

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# COPY bashrc /etc/bash.bashrc
# RUN chmod a+rwx /etc/bash.bashrc

RUN apt-get update \
  # psycopg2 dependencies
  # && apt-get install gcc python3-dev musl-dev \
  && apt-get install -y gcc unzip \
  # && apt-get install -y postgresql postgresql-contrib \
  # && apt-get install postgresql-dev \
  # Pillow dependencies
  && apt-get install -y libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev liblcms2-dev libwebp-dev tk-dev tcl-dev libopenjp2-7-dev pngquant libraqm-dev \
  # CFFI dependencies
  && apt-get install -y libffi-dev python-cairocffi \
  # Translations dependencies
  && apt-get install -y gettext

# Requirements are installed here to ensure they will be cached.
COPY ./requirements /requirements
RUN pip install -r /requirements/production.txt

# PyTorch without CUDA you can change it into what types, you want
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./compose/production/entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint
RUN chmod +x /entrypoint

COPY ./compose/production/start /start
RUN sed -i 's/\r$//g' /start
RUN chmod +x /start


WORKDIR /app

# ENTRYPOINT ["/entrypoint"]
ENTRYPOINT ["/bin/bash"]
