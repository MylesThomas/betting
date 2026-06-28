FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --only-binary :all: forces manylinux wheels — avoids source compilation (AL2 has GCC 7.3.1,
# numpy 2.x requires GCC >= 9.3). fastavro is pure-Python fallback; if no wheel exists it
# compiles fine with older gcc, so allow source for that one package only.
RUN pip install --no-cache-dir --only-binary :all: \
    boto3 \
    "pandas==2.2.3" \
    numpy \
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

# Lambda task root is /var/task
COPY src/nfl_sacks_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
