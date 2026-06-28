FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# scipy is required for nbinom (NegBin inference in run_pipeline.py)
# fastavro: pure-Python fallback for nfl_data_py; build from source if no wheel
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
  && pip install --no-cache-dir fastavro

COPY src/nfl_tackles_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
