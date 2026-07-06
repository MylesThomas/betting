FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --only-binary :all: \
    boto3 \
    "pandas==2.2.3" \
    numpy \
    scipy \
    "pyarrow==17.0.0" \
    "scikit-learn==1.5.2" \
    "joblib==1.4.2" \
    requests \
    pyyaml \
    python-dotenv \
    "nfl-data-py==0.3.2" \
    tqdm \
    "tzdata==2024.1" \
  && pip install --no-cache-dir fastavro \
  && pip install --no-cache-dir xgboost

COPY src/nfl_rec_yards_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY models ${LAMBDA_TASK_ROOT}/models
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
