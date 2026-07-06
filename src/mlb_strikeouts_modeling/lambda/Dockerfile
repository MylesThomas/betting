FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --only-binary :all: \
    boto3 \
    "pandas==2.2.3" \
    numpy \
    "scikit-learn==1.6.1" \
    joblib \
    "pyarrow==17.0.0" \
    requests \
    pyyaml \
    "tzdata==2024.1"

COPY src/mlb_strikeouts_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
