FROM python:3.5-slim
MAINTAINER Qntfy

ENV BUILD_DEPS   \
        autoconf \
        automake \
        libtool  \
        curl     \
        wget     \
        make     \
        g++      \
        gcc      \
        git


RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD . /usr/src/app/

# Install all the things!
RUN set -ex \
        && cd /tmp \
        # Install General Build Dependencies
        && apt-get update -y --no-install-recommends \
        && apt-get upgrade -y --no-install-recommends \
        && apt-get install -y --no-install-recommends ${BUILD_DEPS} \
        && apt-get install -y --no-install-recommends ${RUN_DEPS} \
        # Install the Python dependencies
        && cd /usr/src/app \
		&& pip install --no-cache-dir --upgrade pip \
		&& pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt \
        # Delete build dependencies
		&& rm -rf /tmp/* \
		&& apt-get remove -y ${BUILD_DEPS} \
        && apt-get autoremove -y \
        && apt-get purge -y \
        && apt-get clean -y \
        && rm -rf /var/lib/apt/lists/*

RUN python -m nltk.downloader perluniprops

# Start the server
EXPOSE 5000
CMD ["/usr/src/app/launch.sh"]
