# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY project_name/ project_name/
COPY Makefile Makefile


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e . 

RUN dvc init --no-scm
COPY .dvc .dvc
COPY models.dvc models.dvc
COPY data.dvc data.dvc
RUN dvc config core.no_scm true
RUN make predict
#ENTRYPOINT ["python", "-u", "project_name/predict_model.py" "--model_version", "models/exp1"] 