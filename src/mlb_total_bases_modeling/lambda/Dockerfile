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

# xgboost 2.1.4 ships a py3-none manylinux2014_x86_64 wheel — no compilation needed
RUN pip install --no-cache-dir "xgboost==2.1.4"

# Install pybaseball's required deps manually, skipping matplotlib and its
# transitive deps (pillow, contourpy) which need system libs we don't have.
# We only use statcast() and playerid_reverse_lookup() — neither needs matplotlib.
RUN pip install --no-cache-dir \
    "beautifulsoup4>=4.4.0" \
    "lxml>=4.2.1" \
    "tqdm>=4.50.0" \
    "attrs>=20.3.0" \
    "pygithub>=1.51" && \
    pip install --no-cache-dir --no-deps pybaseball && \
    sed -i '/from .plotting import/d' /var/lang/lib/python3.11/site-packages/pybaseball/__init__.py

COPY src/mlb_total_bases_modeling/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY config ${LAMBDA_TASK_ROOT}/config
COPY .gitignore ${LAMBDA_TASK_ROOT}/.gitignore

CMD ["lambda_function.lambda_handler"]
