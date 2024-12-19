FROM nvcr.io/nvidia/pytorch:24.12-py3
ENV PIP_ROOT_USER_ACTION=ignore

COPY docker/install_deps.sh /usr/local/bin/install_deps.sh
RUN chmod +x /usr/local/bin/install_deps.sh

RUN ["/bin/bash", "-c", "/usr/local/bin/install_deps.sh"]
