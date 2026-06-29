FROM public.ecr.aws/lambda/python:3.13

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Versions pinned to match uv.lock — always run fit_model.py via `uv run`
# to ensure the pickle is saved with the same sklearn version as this image.
RUN pip install --no-cache-dir --only-binary :all: \
    boto3 \
    "pandas==2.3.3" \
    "numpy==2.4.2" \
    "pyarrow==23.0.0" \
    "scikit-learn==1.8.0" \
    "joblib==1.5.3" \
    requests \
    pyyaml \
    python-dotenv \
    "nfl-data-py==0.3.2" \
    tqdm \
    "tzdata==2025.3" \
  && pip install --no-cache-dir fastavro

# Lambda task root is /var/task
COPY src/nfl_sacks_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
