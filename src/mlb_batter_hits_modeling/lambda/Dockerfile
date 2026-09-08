FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --only-binary :all: \
    boto3 \
    "pandas==2.2.3" \
    numpy \
    "scikit-learn==1.7.2" \
    joblib \
    "pyarrow==17.0.0" \
    requests \
    pyyaml \
    "tzdata==2024.1" \
    python-dotenv

# pybaseball for spine updates (used in update_spine.py)
RUN pip install --no-cache-dir \
    "beautifulsoup4>=4.4.0" \
    "lxml>=4.2.1" \
    "tqdm>=4.50.0" \
    "attrs>=20.3.0" \
    "pygithub>=1.51" && \
    pip install --no-cache-dir --no-deps pybaseball && \
    sed -i '/from .plotting import/d' /var/lang/lib/python3.11/site-packages/pybaseball/__init__.py

COPY src/mlb_batter_hits_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
