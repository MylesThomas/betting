FROM python:3.13-slim

ENV PIP_TRUSTED_HOST="pypi.org pypi.python.org files.pythonhosted.org"

RUN pip install awslambdaric --no-cache-dir

COPY requirements_lambda.txt .
RUN pip install -r requirements_lambda.txt --no-cache-dir

COPY lambda_function.py .

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_function.handler"]
