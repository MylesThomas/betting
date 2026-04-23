"""Tests for settlement email plays formatting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.nba_rebounds_settlement_email import (
    format_settlement_email_plays_table,
    format_settlement_email_plays_table_html,
    prepare_email_plays_display_slice,
)


def _sample_row(**kwargs) -> dict:
    base = {
        "player_normalized": "Test Player",
        "strategy_bucket": "both",
        "bookmaker": "fanduel",
        "line": 10.5,
        "reb_actual": 12.0,
        "result": "win",
        "under_odds": -110,
        "date": "2026-04-20",
    }
    base.update(kwargs)
    return base


def test_prepare_email_plays_display_slice_truncation():
    rows = [_sample_row(player_normalized=f"p{i}", result="win") for i in range(5)]
    df = pd.DataFrame(rows)
    slice_df, total, truncated = prepare_email_plays_display_slice(df, max_rows=3)
    assert total == 5
    assert truncated is True
    assert len(slice_df) == 3


def test_format_settlement_email_plays_table_html_contains_headers_and_escaping():
    df = pd.DataFrame(
        [
            _sample_row(
                player_normalized='Evil <script> & "quotes"',
                bookmaker="book&maker",
                result="loss",
            )
        ]
    )
    out = format_settlement_email_plays_table_html(df, max_rows=600)
    assert "<th " in out and ">player</th>" in out
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp;" in out
    assert "book&amp;maker" in out
    assert ">loss</td>" in out


def test_format_settlement_email_plays_table_matches_slice_row_count():
    rows = [_sample_row(player_normalized=f"p{i}") for i in range(4)]
    df = pd.DataFrame(rows)
    text = format_settlement_email_plays_table(df, max_rows=2)
    html_out = format_settlement_email_plays_table_html(df, max_rows=2)
    assert "showing first 2 of 4" in text
    assert "showing first 2 of 4" in html_out
    assert text.count("\n") >= 4
    assert html_out.count("<tr ") == 2


def test_format_settlement_email_plays_table_html_empty():
    assert format_settlement_email_plays_table_html(pd.DataFrame()) == ""
    assert format_settlement_email_plays_table_html(pd.DataFrame(columns=["player_normalized"])) == ""


@pytest.mark.parametrize(
    "result,needle",
    [
        ("win", "#166534"),
        ("loss", "#991b1b"),
        ("push", "#92400e"),
        ("unsettled", "#4b5563"),
    ],
)
def test_result_styling_colors(result: str, needle: str):
    df = pd.DataFrame([_sample_row(result=result)])
    out = format_settlement_email_plays_table_html(df)
    assert needle in out
    assert f">{result}</td>" in out
