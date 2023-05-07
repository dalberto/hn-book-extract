FROM python:3.11-slim

ARG APP_DIR

ENV APP_DIR=$APP_DIR \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.4.2

WORKDIR $APP_DIR

RUN pip install poetry==$POETRY_VERSION

COPY poetry.toml pyproject.toml poetry.lock ./

RUN poetry install

ENTRYPOINT ["python", "extract.py"]
