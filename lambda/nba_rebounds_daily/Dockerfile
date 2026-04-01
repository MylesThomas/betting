FROM public.ecr.aws/lambda/python:3.11

# Keep container behavior predictable.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies required by pipeline + settlement scripts.
RUN pip install --no-cache-dir \
    boto3 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4 \
    xgboost==2.1.1 \
    pyarrow==17.0.0 \
    nba_api \
    pyyaml \
    python-dotenv \
    duckdb==1.1.3 \
    requests

# Lambda task root is /var/task
COPY lambda/nba_rebounds_daily/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY scripts ${LAMBDA_TASK_ROOT}/scripts
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore
COPY workdir_test/rebounds_prod/rebounds_model_features_2025_26.parquet ${LAMBDA_TASK_ROOT}/workdir_test/rebounds_prod/rebounds_model_features_2025_26.parquet

CMD ["lambda_function.lambda_handler"]
