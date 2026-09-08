"""
Scrape historical Reddit posts mentioning Snap/Snapchat using PRAW.

SETUP (one time):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app" → choose "script"
  3. Name: "sentiment-research", redirect uri: http://localhost:8080
  4. Copy client_id (under app name) and client_secret
  5. Set env vars:
       export REDDIT_CLIENT_ID=your_client_id
       export REDDIT_CLIENT_SECRET=your_client_secret

STRATEGY:
  PRAW subreddit.search() returns max 1,000 results per query. We work
  around this by issuing monthly date-windowed queries — each month is
  a separate search with `timestamp:epoch1..epoch2` filter syntax.
  This gives ~full coverage back to 2019 at the cost of more API calls.

Output: data/reddit_snap_raw.parquet
"""

import os
import time
import praw
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

OUT = Path(__file__).parent / "data" / "reddit_snap_raw.parquet"

SUBREDDITS = [
    "cscareerquestions",
    "ExperiencedDevs",
    "layoffs",
    "jobs",
    "softwareengineering",
    "tech",
    "technology",
]

KEYWORDS = ["snapchat", "snap inc", '"snap" layoff', '"snap" job', '"snap" offer']

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 8, 1, tzinfo=timezone.utc)


def month_windows(start: datetime, end: datetime):
    """Yield (window_start_ts, window_end_ts) for each calendar month."""
    cur = start.replace(day=1)
    while cur < end:
        next_month = cur + relativedelta(months=1)
        yield int(cur.timestamp()), int(min(next_month, end).timestamp())
        cur = next_month


def make_reddit() -> praw.Reddit:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit credentials.\n"
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.\n"
            "Create a free app at: https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="sentiment-research/0.1 (educational)",
    )


def search_window(subreddit, keyword: str, ts_start: int, ts_end: int) -> list[dict]:
    """Search one subreddit for one keyword in one time window."""
    # Reddit Lucene syntax for timestamp range
    query = f"{keyword} timestamp:{ts_start}..{ts_end}"
    posts = []
    try:
        results = subreddit.search(
            query,
            sort="new",
            syntax="lucene",
            time_filter="all",
            limit=None,  # fetch up to 1000 (Reddit hard cap per query)
        )
        for post in results:
            posts.append({
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "created_utc": int(post.created_utc),
                "score": post.score,
                "num_comments": post.num_comments,
                "subreddit": post.subreddit.display_name,
                "author": str(post.author) if post.author else "[deleted]",
                "url": post.url,
            })
    except Exception as e:
        print(f"    Error: {e}")
    return posts


def main():
    reddit = make_reddit()

    all_posts: list[dict] = []
    seen_ids: set[str] = set()

    snap_pattern = r'\bsnap\b|\bsnapchat\b|\bsnap inc\b'

    print(f"Scraping Reddit for Snap mentions across {len(SUBREDDITS)} subreddits")
    print(f"Date range: {START.date()} → {END.date()}")
    print(f"Using {len(KEYWORDS)} keyword queries × monthly windows\n")

    for sub_name in SUBREDDITS:
        sub = reddit.subreddit(sub_name)
        sub_posts = 0

        for keyword in KEYWORDS:
            windows = list(month_windows(START, END))
            for ts_start, ts_end in windows:
                month_label = datetime.fromtimestamp(ts_start, tz=timezone.utc).strftime("%Y-%m")
                results = search_window(sub, keyword, ts_start, ts_end)

                new = 0
                for p in results:
                    if p["id"] not in seen_ids:
                        seen_ids.add(p["id"])
                        all_posts.append(p)
                        new += 1
                sub_posts += new

                if new > 0:
                    print(f"  r/{sub_name} | {keyword!r} | {month_label}: +{new}")
                time.sleep(0.6)  # stay well under 60 req/min rate limit

        print(f"  → r/{sub_name} total: {sub_posts} posts\n")

    if not all_posts:
        print("No posts retrieved.")
        return

    df = pd.DataFrame(all_posts)
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["selftext"] = df["selftext"].replace(["[removed]", "[deleted]"], "")

    # Filter to posts that actually mention snap (remove false positives)
    mask = (
        df["title"].str.lower().str.contains(snap_pattern, regex=True, na=False)
        | df["selftext"].str.lower().str.contains(snap_pattern, regex=True, na=False)
    )
    df = df[mask].sort_values("created_utc").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"Saved {len(df):,} posts → {OUT}")
    print(f"\nPosts per subreddit:\n{df['subreddit'].value_counts().to_string()}")
    print(f"\nPosts per year:\n{df['created_dt'].dt.year.value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
