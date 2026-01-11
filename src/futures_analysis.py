"""
Shared Championship Futures Analysis Functions

Purpose:
Core business logic for analyzing championship futures across NFL, NBA, NCAAF, NCAAB.
Extracted from individual sport-specific scripts to eliminate duplication.

Functions:
- get_most_recent_futures_file: Find latest futures CSV
- calculate_vig_by_bookmaker: Calculate vig for each bookmaker
- calculate_fair_probabilities: Remove vig to get true probabilities
- calculate_team_averages: Average odds across bookmakers
- calculate_line_shopping_opportunities: Find best odds per team
- save_analysis_outputs: Save results to CSV

Usage:
    from futures_analysis import calculate_vig_by_bookmaker
    
    vig_df = calculate_vig_by_bookmaker(df)
"""

import pandas as pd
import numpy as np
import boto3
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from odds_utils import odds_to_implied_probability, probability_to_american_odds


def get_most_recent_futures_file(input_dir: Path, file_prefix: str, s3_bucket: str = None) -> Path:
    """
    Find the most recently created futures file in a directory or S3.
    
    Args:
        input_dir: Directory to search (e.g., data/01_input/the-odds-api/nfl/futures)
        file_prefix: Prefix of files to match (e.g., "nfl_super_bowl_futures")
        s3_bucket: Optional S3 bucket to check (e.g., "the-odds-api-mt")
        
    Returns:
        Path to most recent file (downloaded from S3 if needed)
        
    Raises:
        FileNotFoundError: If no matching files found
    """
    # If S3 bucket provided, download from S3
    if s3_bucket:
        try:
            s3_client = boto3.client('s3')
            
            # Determine sport from file prefix
            if 'nfl' in file_prefix or 'super_bowl' in file_prefix:
                s3_prefix = 'nfl/futures/'
            elif 'nba' in file_prefix:
                s3_prefix = 'nba/futures/'
            elif 'ncaaf' in file_prefix:
                s3_prefix = 'ncaaf/futures/'
            elif 'ncaab' in file_prefix:
                s3_prefix = 'ncaab/futures/'
            else:
                raise ValueError(f"Cannot determine sport from file_prefix: {file_prefix}")
            
            # List all files in S3 with the prefix
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                raise FileNotFoundError(f"No files found in s3://{s3_bucket}/{s3_prefix}")
            
            # Filter for files matching the prefix pattern and get most recent
            matching_files = [
                obj for obj in response['Contents']
                if file_prefix in obj['Key'] and obj['Key'].endswith('.csv')
            ]
            
            if not matching_files:
                raise FileNotFoundError(f"No files matching '{file_prefix}' found in s3://{s3_bucket}/{s3_prefix}")
            
            # Sort by last modified (most recent first)
            matching_files.sort(key=lambda x: x['LastModified'], reverse=True)
            most_recent = matching_files[0]
            
            # Download to local temp location
            input_dir.mkdir(parents=True, exist_ok=True)
            local_file = input_dir / Path(most_recent['Key']).name
            
            print(f"📥 Downloading from s3://{s3_bucket}/{most_recent['Key']}")
            s3_client.download_file(s3_bucket, most_recent['Key'], str(local_file))
            print(f"   ✅ Downloaded to {local_file.name}\n")
            
            return local_file
            
        except Exception as e:
            print(f"⚠️  S3 download failed: {e}")
            print(f"   Falling back to local files...\n")
    
    # Check local files
    futures_files = list(input_dir.glob(f'{file_prefix}_*.csv'))
    
    if not futures_files:
        raise FileNotFoundError(f"No futures files found matching: {input_dir}/{file_prefix}_*.csv")
    
    # Sort by modification time (most recent first)
    futures_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return futures_files[0]


def calculate_vig_by_bookmaker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate vig for each bookmaker.
    
    Vig = Total implied probability - 1.0
    Higher vig = worse for bettors
    
    Args:
        df: DataFrame with columns ['bookmaker', 'team', 'implied_prob']
        
    Returns:
        DataFrame with columns ['bookmaker', 'num_teams', 'total_implied_prob', 'vig_pct']
        Sorted by vig_pct descending (worst bookmaker first)
    """
    vig_by_bookmaker = []
    
    for bookmaker in df['bookmaker'].unique():
        bookmaker_df = df[df['bookmaker'] == bookmaker]
        
        total_implied = bookmaker_df['implied_prob'].sum()
        vig_pct = (total_implied - 1.0) * 100
        
        vig_by_bookmaker.append({
            'bookmaker': bookmaker,
            'num_teams': len(bookmaker_df),
            'total_implied_prob': total_implied,
            'vig_pct': vig_pct
        })
    
    vig_df = pd.DataFrame(vig_by_bookmaker)
    
    # Sort by vig descending (worst bookmaker first)
    vig_df = vig_df.sort_values('vig_pct', ascending=False)
    
    return vig_df


def calculate_fair_probabilities(df: pd.DataFrame, bookmaker: str) -> pd.DataFrame:
    """
    Calculate fair probabilities by removing vig from a single bookmaker.
    
    Fair probability = implied probability / sum of all implied probabilities
    This normalizes so probabilities sum to exactly 100%.
    
    Args:
        df: DataFrame with all bookmaker data
        bookmaker: Which bookmaker to use for fair odds calculation
        
    Returns:
        DataFrame with columns ['team', 'odds', 'implied_prob', 'fair_prob', 'fair_odds']
        Sorted by fair_prob descending (best odds first)
    """
    # Filter to single bookmaker
    bookmaker_df = df[df['bookmaker'] == bookmaker].copy()
    
    # Calculate fair probability
    total_implied = bookmaker_df['implied_prob'].sum()
    bookmaker_df['fair_prob'] = bookmaker_df['implied_prob'] / total_implied
    
    # Convert fair probability to American odds
    bookmaker_df['fair_odds'] = bookmaker_df['fair_prob'].apply(
        lambda p: probability_to_american_odds(p * 100)
    )
    
    # Sort by fair probability descending
    bookmaker_df = bookmaker_df.sort_values('fair_prob', ascending=False)
    
    return bookmaker_df[['team', 'odds', 'implied_prob', 'fair_prob', 'fair_odds']]


def calculate_team_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average odds, implied probabilities, and fair odds across all bookmakers.
    
    Also identifies best book and best odds for each team.
    
    Args:
        df: DataFrame with columns ['team', 'bookmaker', 'odds', 'implied_prob', 'record']
        
    Returns:
        DataFrame with one row per team, sorted by fair_prob descending
        Columns: team, implied_prob_avg, implied_prob_min, implied_prob_max, num_books,
                 best_book, best_odds, record, fair_prob, fair_odds, shopping_spread_pct
    """
    # For each team, find which book has the best odds (lowest implied prob)
    best_books = []
    for team in df['team'].unique():
        team_df = df[df['team'] == team]
        best_idx = team_df['implied_prob'].idxmin()
        best_book = df.loc[best_idx, 'bookmaker']
        best_odds = df.loc[best_idx, 'odds']
        best_books.append({
            'team': team,
            'best_book': best_book,
            'best_odds': best_odds
        })
    
    best_books_df = pd.DataFrame(best_books)
    
    # Get team records (take first record for each team since they're all the same)
    if 'record' in df.columns:
        team_records = df.groupby('team')['record'].first().reset_index()
    else:
        team_records = None
    
    # Group by team and calculate averages, min, max
    team_avg = df.groupby('team').agg({
        'implied_prob': ['mean', 'min', 'max'],
        'odds': 'count'
    })
    
    # Flatten column names
    team_avg.columns = ['implied_prob_avg', 'implied_prob_min', 'implied_prob_max', 'num_books']
    team_avg = team_avg.reset_index()
    
    # Merge best book info
    team_avg = team_avg.merge(best_books_df, on='team')
    
    # Merge team records if available
    if team_records is not None:
        team_avg = team_avg.merge(team_records, on='team', how='left')
        team_avg['record'] = team_avg['record'].fillna('-')
    else:
        team_avg['record'] = '-'
    
    # Calculate fair probability (remove vig)
    # Total implied prob across all teams (average across books)
    total_implied_avg = team_avg['implied_prob_avg'].sum()
    
    # Fair probability = normalize to sum to 1.0
    team_avg['fair_prob'] = team_avg['implied_prob_avg'] / total_implied_avg
    
    # Convert fair probability to American odds
    team_avg['fair_odds'] = team_avg['fair_prob'].apply(
        lambda p: probability_to_american_odds(p * 100)
    )
    
    # Calculate line shopping opportunity (spread between min and max)
    team_avg['shopping_spread_pct'] = (team_avg['implied_prob_max'] - team_avg['implied_prob_min']) * 100
    
    # Sort by fair_prob desc (best championship odds first)
    team_avg = team_avg.sort_values('fair_prob', ascending=False)
    
    return team_avg


def save_analysis_outputs(
    team_avg: pd.DataFrame, 
    vig_df: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
    save_locally: bool = False,
    s3_bucket: str = None,
    s3_path: str = None
) -> Tuple[Path, Path]:
    """
    Save analysis results to CSV files (local and/or S3).
    
    Args:
        team_avg: Team averages DataFrame
        vig_df: Bookmaker vig DataFrame
        output_dir: Directory to save files (e.g., data/04_output/nfl)
        output_prefix: Prefix for output files (e.g., "nfl_championship")
        save_locally: If True, save to local filesystem
        s3_bucket: S3 bucket name for output (e.g., "nfl-betting-mt")
        s3_path: S3 path prefix (e.g., "analysis")
        
    Returns:
        Tuple of (team_averages_path, metadata_path)
    """
    # Prepare files
    team_avg_file = output_dir / f'{output_prefix}_fair_odds.csv'
    metadata_file = output_dir / f'{output_prefix}_metadata.csv'
    
    # Calculate average vig across bookmakers
    avg_vig_pct = vig_df['vig_pct'].mean()
    
    metadata = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_bookmakers': len(vig_df),
        'num_teams': len(team_avg),
        'avg_vig_pct': avg_vig_pct,
        'min_vig_pct': vig_df['vig_pct'].min(),
        'max_vig_pct': vig_df['vig_pct'].max(),
        'best_bookmaker': vig_df.iloc[-1]['bookmaker'],  # Last = lowest vig
        'worst_bookmaker': vig_df.iloc[0]['bookmaker'],  # First = highest vig
    }])
    
    # Save locally if requested
    if save_locally:
        output_dir.mkdir(parents=True, exist_ok=True)
        team_avg.to_csv(team_avg_file, index=False)
        metadata.to_csv(metadata_file, index=False)
    
    # Save to S3 if bucket provided
    if s3_bucket and s3_path:
        try:
            s3_client = boto3.client('s3')
            
            # Upload team averages
            team_avg_s3_key = f"{s3_path}/{output_prefix}_fair_odds.csv"
            team_avg_csv = team_avg.to_csv(index=False)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=team_avg_s3_key,
                Body=team_avg_csv,
                ContentType='text/csv'
            )
            print(f"☁️  Uploaded to s3://{s3_bucket}/{team_avg_s3_key}")
            
            # Upload metadata
            metadata_s3_key = f"{s3_path}/{output_prefix}_metadata.csv"
            metadata_csv = metadata.to_csv(index=False)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=metadata_s3_key,
                Body=metadata_csv,
                ContentType='text/csv'
            )
            print(f"☁️  Uploaded to s3://{s3_bucket}/{metadata_s3_key}")
            
        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")
    
    return team_avg_file, metadata_file

