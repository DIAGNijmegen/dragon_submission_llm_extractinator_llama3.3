FROM --platform=linux/amd64 python:3.12-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN python -m pip install --user dragon_baseline==0.4.5

COPY --chown=user:user . /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]