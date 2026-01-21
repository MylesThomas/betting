"""
S3 utility functions for uploading/downloading data files.

Context:
Helper functions to interact with S3 buckets for storing betting data.
Keeps scripts clean and centralizes S3 logic.
"""

import boto3
import pandas as pd
from pathlib import Path
from io import StringIO, BytesIO
import os


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client('s3')


def upload_df_to_s3(df, bucket, s3_key):
    """
    Upload DataFrame to S3 as CSV.
    
    Args:
        df: pandas DataFrame
        bucket: S3 bucket name
        s3_key: S3 object key (path)
    
    Returns:
        S3 URI (s3://bucket/key)
    """
    s3 = get_s3_client()
    
    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    
    return f"s3://{bucket}/{s3_key}"


def read_df_from_s3(bucket, s3_key):
    """
    Read CSV from S3 into DataFrame.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 object key (path)
    
    Returns:
        pandas DataFrame
    """
    s3 = get_s3_client()
    
    # Download from S3
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    
    # Read into DataFrame
    df = pd.read_csv(BytesIO(obj['Body'].read()))
    
    return df


def list_s3_files(bucket, prefix):
    """
    List files in S3 bucket with given prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
    
    Returns:
        List of S3 keys
    """
    s3 = get_s3_client()
    
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        return []
    
    return [obj['Key'] for obj in response['Contents']]


def get_latest_file_from_s3(bucket, prefix, pattern='*.csv'):
    """
    Get the most recent file from S3 based on LastModified.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        pattern: File pattern (default: '*.csv')
    
    Returns:
        S3 key of most recent file, or None if no files found
    """
    s3 = get_s3_client()
    
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        return None
    
    # Filter by pattern if needed
    files = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
    
    if not files:
        return None
    
    # Sort by LastModified and get the most recent
    latest_file = max(files, key=lambda x: x['LastModified'])
    
    return latest_file['Key']
