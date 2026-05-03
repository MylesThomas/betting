"""S3/local transparent IO helpers shared across the rebounds pipeline."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path


def _is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _s3_parts(uri: str) -> tuple[str, str]:
    rest = str(uri)[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _s3():
    import boto3
    return boto3.client("s3")


def read_parquet_any(uri: str, **kwargs):
    import pandas as pd
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        buf = BytesIO(_s3().get_object(Bucket=bucket, Key=key)["Body"].read())
        return pd.read_parquet(buf, **kwargs)
    return pd.read_parquet(Path(uri).expanduser(), **kwargs)


def write_parquet_any(df, uri: str) -> None:
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def read_csv_any(uri: str, **kwargs):
    import pandas as pd
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        buf = BytesIO(_s3().get_object(Bucket=bucket, Key=key)["Body"].read())
        return pd.read_csv(buf, **kwargs)
    return pd.read_csv(Path(uri).expanduser(), **kwargs)


def write_csv_str(content: str, uri: str) -> None:
    """Write a CSV string (not DataFrame) to local or S3."""
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        _s3().put_object(Bucket=bucket, Key=key, Body=content.encode())
    else:
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def read_yaml_any(uri: str) -> dict:
    import yaml
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        return yaml.safe_load(body)
    with open(Path(uri).expanduser(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json_any(uri: str) -> dict:
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        return json.loads(body)
    with open(Path(uri).expanduser(), encoding="utf-8") as f:
        return json.load(f)


def write_json_any(obj: dict, uri: str) -> None:
    content = json.dumps(obj, indent=2)
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        _s3().put_object(Bucket=bucket, Key=key, Body=content.encode())
    else:
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def read_bytes_any(uri: str) -> bytes:
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).expanduser().read_bytes()


def write_bytes_any(data: bytes, uri: str) -> None:
    if _is_s3(uri):
        bucket, key = _s3_parts(uri)
        _s3().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def uri_exists(uri: str) -> bool:
    if _is_s3(uri):
        import botocore.exceptions
        bucket, key = _s3_parts(uri)
        try:
            _s3().head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False
    return Path(uri).expanduser().exists()
