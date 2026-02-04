"""
Get Lambda Function Logs

Interactive script to fetch and display the most recent logs from any Lambda function.

Usage:
    python tmp/get_lambda_logs.py
    python tmp/get_lambda_logs.py --lambda-function-name my-function

Author: Myles Thomas
Date: 2026-02-03
"""

import argparse
import boto3
from botocore.exceptions import ClientError
from datetime import datetime


# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'info': 'ℹ️',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'search': '🔍',
    'logs': '📋',
    'calendar': '📅',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_aws_region() -> str:
    """
    Get current AWS region from session.
    
    Returns:
        AWS region string
    """
    session = boto3.session.Session()
    region = session.region_name
    
    if not region:
        # Fallback to default region
        region = 'us-east-2'
    
    return region


def get_latest_log_stream(logs_client, log_group: str) -> str:
    """
    Get the most recent log stream for a log group.
    
    Args:
        logs_client: Boto3 CloudWatch Logs client
        log_group: Log group name
    
    Returns:
        Log stream name or None if not found
    """
    try:
        response = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )
        
        if response['logStreams']:
            return response['logStreams'][0]['logStreamName']
        
        return None
    
    except ClientError as e:
        print(f"{EMOJI['error']} Error getting log stream: {e}")
        return None


def format_timestamp(timestamp_ms: int) -> str:
    """
    Format timestamp from milliseconds to readable string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
    
    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def get_log_events(logs_client, log_group: str, log_stream: str) -> list:
    """
    Get all log events from a log stream.
    
    Args:
        logs_client: Boto3 CloudWatch Logs client
        log_group: Log group name
        log_stream: Log stream name
    
    Returns:
        List of log event dictionaries
    """
    try:
        events = []
        next_token = None
        
        while True:
            if next_token:
                response = logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=log_stream,
                    startFromHead=True,
                    nextToken=next_token
                )
            else:
                response = logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=log_stream,
                    startFromHead=True
                )
            
            events.extend(response['events'])
            
            # Check if there are more events
            if 'nextForwardToken' in response and response['nextForwardToken'] != next_token:
                next_token = response['nextForwardToken']
            else:
                break
        
        return events
    
    except ClientError as e:
        print(f"{EMOJI['error']} Error getting log events: {e}")
        return []


def print_logs(events: list, show_timestamps: bool = False) -> None:
    """
    Print log events to console.
    
    Args:
        events: List of log event dictionaries
        show_timestamps: Whether to show timestamps
    """
    if not events:
        print(f"{EMOJI['warning']} No log events found")
        return
    
    print(f"\n{EMOJI['logs']} Log Output ({len(events)} events):")
    print("="*80)
    
    for event in events:
        message = event['message'].rstrip()
        
        if show_timestamps:
            timestamp = format_timestamp(event['timestamp'])
            print(f"[{timestamp}] {message}")
        else:
            print(message)
    
    print("="*80)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for interactive log fetching."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Fetch and display the most recent logs from a Lambda function'
    )
    parser.add_argument(
        '--lambda-function-name',
        type=str,
        help='Name of the Lambda function (skips interactive prompt)'
    )
    args = parser.parse_args()
    
    # Step 1: Display AWS credentials info
    print("\n" + "="*80)
    print(f"{EMOJI['info']} AWS Configuration")
    print("="*80)
    
    region = get_aws_region()
    print(f"Region: {region}")
    
    # Get AWS account info if possible
    try:
        sts_client = boto3.client('sts', region_name=region)
        identity = sts_client.get_caller_identity()
        print(f"Account: {identity['Account']}")
        print(f"User/Role: {identity['Arn'].split('/')[-1]}")
    except Exception:
        pass
    
    print("="*80 + "\n")
    
    # Step 2: Get Lambda function name from args or prompt
    if args.lambda_function_name:
        lambda_function = args.lambda_function_name.strip()
    else:
        lambda_function = input(f"{EMOJI['search']} Enter Lambda function name: ").strip()
    
    if not lambda_function:
        print(f"{EMOJI['error']} No function name provided")
        return
    
    print(f"\n{EMOJI['info']} Fetching logs for: {lambda_function}")
    
    # Step 3: Construct log group name (Lambda convention)
    log_group = f'/aws/lambda/{lambda_function}'
    
    # Step 4: Get logs
    logs_client = boto3.client('logs', region_name=region)
    
    print(f"{EMOJI['info']} Looking for log group: {log_group}")
    
    # Get latest log stream
    log_stream = get_latest_log_stream(logs_client, log_group)
    
    if not log_stream:
        print(f"{EMOJI['error']} No log streams found for this function")
        print(f"\nPossible reasons:")
        print(f"  1. Function hasn't been invoked yet")
        print(f"  2. Function name is incorrect")
        print(f"  3. Region is incorrect (currently using: {region})")
        return
    
    print(f"{EMOJI['success']} Found latest log stream: {log_stream}")
    
    # Get all events from the stream
    print(f"{EMOJI['info']} Fetching log events...")
    events = get_log_events(logs_client, log_group, log_stream)
    
    # Print logs
    print_logs(events, show_timestamps=False)
    
    # Show summary
    if events:
        first_time = format_timestamp(events[0]['timestamp'])
        last_time = format_timestamp(events[-1]['timestamp'])
        
        # Calculate duration
        first_ms = events[0]['timestamp']
        last_ms = events[-1]['timestamp']
        duration_ms = last_ms - first_ms
        duration_seconds = duration_ms / 1000
        duration_minutes = duration_seconds / 60
        
        print(f"\n{EMOJI['calendar']} Log Stream Time Range:")
        print(f"  First: {first_time}")
        print(f"  Last:  {last_time}")
        
        # Show duration in appropriate units
        if duration_minutes >= 1:
            print(f"  Duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")
        else:
            print(f"  Duration: {duration_seconds:.2f} seconds ({duration_ms} ms)")


if __name__ == '__main__':
    main()
