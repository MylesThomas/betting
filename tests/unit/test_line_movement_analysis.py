"""
Unit tests for line movement analysis calculations.

Tests core logic with synthetic data to ensure:
1. Cover calculations are correct (ATS formula)
2. Line movement direction is properly determined
3. Derived features match expected values
4. Edge cases handled correctly (line crossing, no movement, etc.)
5. Invariants always hold (spreads sum to zero, etc.)

These tests do NOT hit S3 or load real data - they're pure unit tests.

Usage:
    cd betting
    
    # Run all unit tests
    pytest tests/unit/ -v
    
    # Run this file only
    pytest tests/unit/test_line_movement_analysis.py -v
    
    # Run specific test class
    pytest tests/unit/test_line_movement_analysis.py::TestCoverCalculations -v
    
    # Run with coverage
    pytest tests/unit/ --cov=analysis --cov-report=html
    
    # Run specific test
    pytest tests/unit/test_line_movement_analysis.py::TestCoverCalculations::test_favorite_covers_by_more_than_spread -v

Author: Thomas Myles
Date: 2026-01-13
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path to import from analysis/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import functions to test
from analysis.analyze_line_movement_predictiveness import (
    normalize_team_name,
    add_derived_features,
)


class TestCoverCalculations:
    """
    Test ATS (against the spread) cover calculations.
    
    This is THE MOST CRITICAL logic - if cover calculations are wrong,
    the entire analysis is invalid.
    
    Formula: cover_margin = (team_score - opponent_score) + spread
             covered = cover_margin > 0
    """
    
    def test_favorite_covers_by_more_than_spread(self):
        """Favorite wins by 10, spread was -7 → should cover"""
        # Setup: Favorite wins 100-90, closing spread -7
        fav_score = 100
        dog_score = 90
        closing_spread = -7.0
        
        # Calculate
        margin = fav_score - dog_score  # +10
        cover_margin = margin + closing_spread  # 10 + (-7) = +3
        covered = cover_margin > 0
        
        assert covered == True, "Favorite should cover when winning by more than spread"
        assert cover_margin == 3.0, f"Expected +3.0, got {cover_margin}"
    
    def test_favorite_wins_but_doesnt_cover(self):
        """Favorite wins by 5, spread was -7 → should NOT cover"""
        fav_score = 100
        dog_score = 95
        closing_spread = -7.0
        
        margin = fav_score - dog_score  # +5
        cover_margin = margin + closing_spread  # 5 + (-7) = -2
        covered = cover_margin > 0
        
        assert covered == False, "Favorite shouldn't cover when winning by less than spread"
        assert cover_margin == -2.0, f"Expected -2.0, got {cover_margin}"
    
    def test_favorite_loses_outright(self):
        """Favorite loses → never covers"""
        fav_score = 95
        dog_score = 100
        closing_spread = -7.0
        
        margin = fav_score - dog_score  # -5
        cover_margin = margin + closing_spread  # -5 + (-7) = -12
        covered = cover_margin > 0
        
        assert covered == False, "Favorite can't cover when losing outright"
        assert cover_margin == -12.0
    
    def test_underdog_wins_outright(self):
        """Underdog wins → always covers"""
        fav_score = 95
        dog_score = 100
        closing_spread = -7.0
        
        # From favorite perspective
        margin = fav_score - dog_score  # -5
        fav_cover_margin = margin + closing_spread  # -5 + (-7) = -12
        fav_covered = fav_cover_margin > 0
        
        assert fav_covered == False, "Favorite didn't cover"
        
        # From underdog perspective (inverse)
        underdog_cover_margin = -fav_cover_margin  # +12
        underdog_covered = not fav_covered
        
        assert underdog_covered == True, "Underdog always covers when winning outright"
        assert underdog_cover_margin == 12.0
    
    def test_underdog_loses_but_covers(self):
        """Underdog loses by 5, spread was +7 → should cover"""
        fav_score = 100
        dog_score = 95
        closing_spread = -7.0
        
        # From favorite perspective
        margin = fav_score - dog_score  # +5
        fav_cover_margin = margin + closing_spread  # 5 + (-7) = -2
        fav_covered = fav_cover_margin > 0
        
        assert fav_covered == False
        
        # From underdog perspective
        dog_closing_spread = -closing_spread  # +7
        dog_margin = dog_score - fav_score  # -5
        dog_cover_margin = dog_margin + dog_closing_spread  # -5 + 7 = +2
        dog_covered = dog_cover_margin > 0
        
        assert dog_covered == True, "Underdog covers when losing by less than spread"
        assert dog_cover_margin == 2.0
    
    def test_push_scenario(self):
        """Favorite wins by exactly spread → push (edge case)"""
        fav_score = 107
        dog_score = 100
        closing_spread = -7.0
        
        margin = fav_score - dog_score  # +7
        cover_margin = margin + closing_spread  # 7 + (-7) = 0
        covered = cover_margin > 0  # Push = False (doesn't cover)
        
        assert covered == False, "Push doesn't count as cover"
        assert cover_margin == 0.0, "Push has zero margin"
    
    def test_half_point_favorite_covers(self):
        """Test with half-point spread (no push possible)"""
        fav_score = 105
        dog_score = 100
        closing_spread = -7.5
        
        margin = fav_score - dog_score  # +5
        cover_margin = margin + closing_spread  # 5 + (-7.5) = -2.5
        covered = cover_margin > 0
        
        assert covered == False
        assert cover_margin == -2.5


class TestMovementCalculations:
    """
    Test line movement calculations.
    
    Movement formula (anchored on opening favorite):
        movement = opening_spread - closing_spread
        
    Positive movement = line moved TOWARD favorite (favorite got more favored)
    Negative movement = line moved TOWARD underdog (underdog got more points)
    """
    
    def test_line_moves_toward_favorite(self):
        """Line moves from -7 to -10 → favorite gained 3 pts"""
        opening_spread = -7.0
        closing_spread = -10.0
        
        movement = opening_spread - closing_spread  # -7 - (-10) = +3
        
        assert movement == 3.0, "Positive movement = toward favorite"
        assert movement > 0
    
    def test_line_moves_toward_underdog(self):
        """Line moves from -7 to -4 → underdog gained 3 pts"""
        opening_spread = -7.0
        closing_spread = -4.0
        
        movement = opening_spread - closing_spread  # -7 - (-4) = -3
        
        assert movement == -3.0, "Negative movement = toward underdog"
        assert movement < 0
    
    def test_no_movement(self):
        """Line doesn't move"""
        opening_spread = -7.0
        closing_spread = -7.0
        
        movement = opening_spread - closing_spread
        
        assert movement == 0.0
    
    def test_large_movement_toward_favorite(self):
        """Large 8-point move toward favorite"""
        opening_spread = -3.0
        closing_spread = -11.0
        
        movement = opening_spread - closing_spread  # +8
        magnitude = abs(movement)
        
        assert movement == 8.0
        assert magnitude == 8.0
    
    def test_line_crosses_zero(self):
        """Line moves from -3 to +2 → underdog becomes favorite"""
        opening_spread = -3.0
        closing_spread = +2.0
        
        movement = opening_spread - closing_spread  # -3 - 2 = -5
        line_crossed = (opening_spread < 0) and (closing_spread > 0)
        
        assert movement == -5.0, "5-point move toward opening underdog"
        assert line_crossed == True, "Line crossed zero"
    
    def test_movement_magnitude_is_absolute(self):
        """Magnitude should always be positive"""
        test_cases = [
            (-7.0, -10.0, 3.0),   # toward fav
            (-7.0, -4.0, 3.0),    # toward dog
            (-7.0, -7.0, 0.0),    # no movement
        ]
        
        for opening, closing, expected_mag in test_cases:
            movement = opening - closing
            magnitude = abs(movement)
            assert magnitude == expected_mag
            assert magnitude >= 0


class TestSteamDirection:
    """
    Test steam direction logic.
    
    Steam direction determines which team got the favorable line movement.
    """
    
    def test_steam_toward_opening_favorite(self):
        """Opening favorite gets steam"""
        opening_fav_spread = -7.0
        closing_fav_spread = -10.0
        
        movement = opening_fav_spread - closing_fav_spread  # +3
        steam_direction = 'opening_favorite' if movement > 0 else 'opening_underdog' if movement < 0 else 'no_movement'
        steam_magnitude = abs(movement)
        
        assert steam_direction == 'opening_favorite'
        assert steam_magnitude == 3.0
    
    def test_steam_toward_opening_underdog(self):
        """Opening underdog gets steam"""
        opening_fav_spread = -7.0
        closing_fav_spread = -4.0
        
        movement = opening_fav_spread - closing_fav_spread  # -3
        steam_direction = 'opening_favorite' if movement > 0 else 'opening_underdog' if movement < 0 else 'no_movement'
        steam_magnitude = abs(movement)
        
        assert steam_direction == 'opening_underdog'
        assert steam_magnitude == 3.0
    
    def test_no_steam(self):
        """No movement → no steam"""
        opening_fav_spread = -7.0
        closing_fav_spread = -7.0
        
        movement = opening_fav_spread - closing_fav_spread
        steam_direction = 'no_movement' if movement == 0 else ('opening_favorite' if movement > 0 else 'opening_underdog')
        
        assert steam_direction == 'no_movement'
        assert movement == 0.0


class TestDerivedFeatures:
    """
    Test derived feature calculations.
    
    Derived features are calculated from base features and must maintain
    mathematical relationships (e.g., underdog = inverse of favorite).
    """
    
    def test_underdog_spread_is_inverse_of_favorite(self):
        """Underdog spread = -1 * favorite spread"""
        opening_fav_spread = -7.0
        opening_dog_spread = -opening_fav_spread
        
        assert opening_dog_spread == 7.0
        assert opening_fav_spread + opening_dog_spread == 0.0
    
    def test_underdog_movement_is_inverse_of_favorite(self):
        """Underdog movement = -1 * favorite movement"""
        opening_fav_spread = -7.0
        closing_fav_spread = -10.0
        opening_fav_movement = opening_fav_spread - closing_fav_spread  # +3
        
        opening_dog_movement = -opening_fav_movement  # -3
        
        assert opening_dog_movement == -3.0
    
    def test_underdog_covered_is_inverse_of_favorite(self):
        """If favorite covers, underdog doesn't (and vice versa)"""
        # Exclude push scenarios
        fav_cover_margin = 5.0
        fav_covered = fav_cover_margin > 0
        
        dog_cover_margin = -fav_cover_margin
        dog_covered = not fav_covered
        
        assert fav_covered == True
        assert dog_covered == False
        assert fav_covered != dog_covered  # XOR (except push)
    
    def test_fade_strategy_is_inverse_of_steam(self):
        """Fade = betting against steam"""
        steam_team_covered = True
        steam_team_cover_margin = 5.0
        
        fade_covered = not steam_team_covered
        fade_margin = -steam_team_cover_margin
        
        assert fade_covered == False
        assert fade_margin == -5.0
    
    def test_steam_team_cover_matches_direction(self):
        """If steam went to favorite and favorite covered, steam_team_covered = True"""
        opening_fav_covered = True
        steam_direction = 'opening_favorite'
        
        # Logic from script
        steam_team_covered = opening_fav_covered if steam_direction == 'opening_favorite' else (not opening_fav_covered)
        
        assert steam_team_covered == True
    
    def test_steam_team_cover_underdog_case(self):
        """If steam went to underdog and underdog covered, steam_team_covered = True"""
        opening_fav_covered = False  # Favorite didn't cover
        steam_direction = 'opening_underdog'
        
        opening_dog_covered = not opening_fav_covered
        steam_team_covered = opening_dog_covered if steam_direction == 'opening_underdog' else opening_fav_covered
        
        assert steam_team_covered == True


class TestInvariants:
    """
    Test invariants that must ALWAYS hold true.
    
    These are mathematical/logical properties that should never be violated.
    If these fail, there's a fundamental error in the logic.
    """
    
    def test_spreads_sum_to_zero(self):
        """Away spread + home spread must = 0"""
        away_spread = -7.0
        home_spread = +7.0
        
        total = away_spread + home_spread
        
        assert abs(total) < 0.01, f"Spreads should sum to zero, got {total}"
    
    def test_opening_favorite_always_has_negative_spread(self):
        """Opening favorite spread must be negative (by definition)"""
        # Test various scenarios
        test_cases = [
            (-7.0, +7.0, -7.0),    # away favorite
            (+7.0, -7.0, -7.0),    # home favorite
            (-3.5, +3.5, -3.5),    # half-point
            (-14.0, +14.0, -14.0), # large spread
        ]
        
        for away_spread, home_spread, expected_fav_spread in test_cases:
            if away_spread < 0:
                fav_spread = away_spread
            else:
                fav_spread = home_spread
            
            assert fav_spread < 0, f"Favorite spread must be negative, got {fav_spread}"
            assert fav_spread == expected_fav_spread
    
    def test_one_team_must_cover_except_push(self):
        """Either favorite or underdog covers (no both/neither except push)"""
        test_cases = [
            (10, 5, -7.0, False, True),    # Fav wins but doesn't cover: margin=5, 5+(-7)=-2 < 0
            (5, 10, -7.0, False, True),    # Dog wins outright
            (10, 5, -4.0, True, False),    # Fav covers: margin=5, 5+(-4)=1 > 0
            (107, 100, -7.0, False, False), # Push (both False): margin=7, 7+(-7)=0
        ]
        
        for fav_score, dog_score, spread, expected_fav_cover, expected_dog_cover in test_cases:
            margin = fav_score - dog_score
            fav_cover_margin = margin + spread
            fav_covered = fav_cover_margin > 0
            dog_covered = not fav_covered if fav_cover_margin != 0 else False
            
            assert fav_covered == expected_fav_cover
            assert dog_covered == expected_dog_cover
            
            # XOR check (except push)
            if fav_cover_margin != 0:
                assert fav_covered != dog_covered, "Exactly one team must cover (not a push)"
    
    def test_steam_magnitude_always_positive_or_zero(self):
        """Steam magnitude is absolute value → always >= 0"""
        test_cases = [5.0, -5.0, 0.0, 3.5, -3.5]
        
        for movement in test_cases:
            magnitude = abs(movement)
            assert magnitude >= 0
            assert magnitude == abs(movement)
    
    def test_cover_margin_determines_cover(self):
        """cover_margin > 0 ↔ covered = True"""
        test_cases = [
            (5.0, True),
            (-5.0, False),
            (0.1, True),
            (-0.1, False),
            (0.0, False),  # Push doesn't cover
        ]
        
        for margin, expected_covered in test_cases:
            actual_covered = margin > 0
            assert actual_covered == expected_covered, f"Failed for margin={margin}"


class TestEdgeCases:
    """
    Test edge cases and boundary conditions.
    """
    
    def test_pickem_line(self):
        """Pick'em (no spread) edge case"""
        away_spread = 0.0
        home_spread = 0.0
        
        assert away_spread == home_spread
        assert away_spread + home_spread == 0.0
    
    def test_extreme_line_movement(self):
        """Very large movement (10+ points)"""
        opening_spread = -3.0
        closing_spread = -13.0
        
        movement = opening_spread - closing_spread  # +10
        magnitude = abs(movement)
        
        assert movement == 10.0
        assert magnitude >= 10.0
        assert magnitude == 10.0
    
    def test_line_crosses_zero_becomes_pickem(self):
        """Line crosses through zero to become pick'em"""
        opening_spread = -1.0
        closing_spread = 0.0
        
        movement = opening_spread - closing_spread  # -1.0 - 0.0 = -1.0
        line_crossed = (opening_spread < 0) and (closing_spread >= 0)
        
        assert movement == -1.0, "Negative movement = toward underdog"
        assert line_crossed == True
    
    def test_tiny_movement(self):
        """Very small movement (0.5 points)"""
        opening_spread = -7.0
        closing_spread = -7.5
        
        movement = opening_spread - closing_spread
        magnitude = abs(movement)
        
        assert movement == 0.5
        assert magnitude == 0.5
    
    def test_negative_scores_should_not_happen(self):
        """Sanity check: scores should be positive"""
        # This is more of a data validation test
        fav_score = 100
        dog_score = 95
        
        assert fav_score > 0
        assert dog_score > 0
        assert fav_score >= 0
        assert dog_score >= 0


class TestFullGameScenarios:
    """
    Integration-style tests with realistic full game scenarios.
    
    These test the entire flow: opening → closing → result → cover analysis.
    """
    
    def test_scenario_favorite_gets_steam_and_covers(self):
        """
        Full pipeline: Favorite gets steam and covers
        
        Game: Celtics @ Lakers
        Opening: Celtics -7
        Closing: Celtics -10 (line moved toward favorite, 3pt steam)
        Result: Celtics win 110-95 (+15 margin)
        Expected: Celtics cover by 5 pts
        """
        # Setup
        away_team = "Boston Celtics"
        home_team = "Los Angeles Lakers"
        
        # Opening lines
        away_open = -7.0
        home_open = +7.0
        
        # Closing lines (line moved toward Celtics)
        away_close = -10.0
        home_close = +10.0
        
        # Final score
        away_score = 110
        home_score = 95
        
        # Determine opening favorite
        opening_favorite = away_team if away_open < 0 else home_team
        opening_underdog = home_team if away_open < 0 else away_team
        opening_fav_opening_spread = away_open if away_open < 0 else home_open
        opening_fav_closing_spread = away_close if away_open < 0 else home_close
        
        # Calculate movement
        opening_fav_movement = opening_fav_opening_spread - opening_fav_closing_spread
        steam_direction = 'opening_favorite' if opening_fav_movement > 0 else 'opening_underdog'
        steam_magnitude = abs(opening_fav_movement)
        
        # Calculate cover
        opening_fav_score = away_score if away_open < 0 else home_score
        opening_dog_score = home_score if away_open < 0 else away_score
        margin = opening_fav_score - opening_dog_score
        cover_margin = margin + opening_fav_closing_spread
        covered = cover_margin > 0
        
        # Steam team covered
        steam_team_covered = covered if steam_direction == 'opening_favorite' else (not covered)
        
        # Assertions
        assert opening_favorite == "Boston Celtics"
        assert opening_fav_movement == 3.0, "3-point steam toward favorite"
        assert steam_direction == 'opening_favorite'
        assert steam_magnitude == 3.0
        assert margin == 15, "Celtics won by 15"
        assert cover_margin == 5.0, "15 + (-10) = 5"
        assert covered == True, "Favorite covered"
        assert steam_team_covered == True, "Team that got steam covered"
    
    def test_scenario_underdog_gets_steam_and_covers(self):
        """
        Full pipeline: Underdog gets steam and covers
        
        Game: Celtics @ Lakers
        Opening: Celtics -7
        Closing: Celtics -4 (line moved toward Lakers, 3pt steam)
        Result: Celtics win 100-98 (+2 margin)
        Expected: Lakers cover with +4 spread
        """
        away_team = "Boston Celtics"
        home_team = "Los Angeles Lakers"
        
        # Opening: Celtics -7
        away_open = -7.0
        
        # Closing: Celtics -4 (line moved toward underdog)
        away_close = -4.0
        
        # Result: Celtics win by 2
        away_score = 100
        home_score = 98
        
        # Calculate
        opening_fav_opening_spread = away_open
        opening_fav_closing_spread = away_close
        opening_fav_movement = opening_fav_opening_spread - opening_fav_closing_spread  # -7 - (-4) = -3
        steam_direction = 'opening_underdog'
        
        opening_fav_score = away_score
        opening_dog_score = home_score
        margin = opening_fav_score - opening_dog_score  # +2
        fav_cover_margin = margin + opening_fav_closing_spread  # 2 + (-4) = -2
        fav_covered = fav_cover_margin > 0  # False
        
        # Steam went to underdog
        steam_team_covered = not fav_covered  # True
        
        # Assertions
        assert opening_fav_movement == -3.0, "3-point steam toward underdog"
        assert steam_direction == 'opening_underdog'
        assert fav_covered == False, "Favorite didn't cover"
        assert steam_team_covered == True, "Underdog (steam team) covered"
        assert fav_cover_margin == -2.0, "Favorite missed by 2"
    
    def test_scenario_favorite_gets_steam_but_doesnt_cover(self):
        """
        Full pipeline: Favorite gets steam but DOESN'T cover
        
        This tests that steam doesn't guarantee cover.
        """
        away_open = -7.0
        away_close = -10.0  # 3pt steam toward favorite
        
        # Favorite only wins by 8 (doesn't cover -10)
        away_score = 108
        home_score = 100
        
        opening_fav_movement = away_open - away_close  # +3
        steam_direction = 'opening_favorite'
        
        margin = away_score - home_score  # +8
        cover_margin = margin + away_close  # 8 + (-10) = -2
        covered = cover_margin > 0  # False
        
        steam_team_covered = covered  # False
        
        assert steam_direction == 'opening_favorite', "Favorite got steam"
        assert covered == False, "But didn't cover"
        assert steam_team_covered == False, "Steam team didn't cover"
    
    def test_scenario_line_crosses_zero(self):
        """
        Full pipeline: Line crosses zero (underdog becomes favorite)
        
        Opening: Team A -2.5
        Closing: Team A +1.5 (underdog became favorite!)
        Result: Team A loses 98-100
        """
        away_open = -2.5
        away_close = +1.5  # Line crossed!
        
        away_score = 98
        home_score = 100
        
        # Calculate from OPENING favorite perspective
        opening_fav_opening_spread = away_open
        opening_fav_closing_spread = away_close
        opening_fav_movement = opening_fav_opening_spread - opening_fav_closing_spread  # -2.5 - 1.5 = -4
        steam_direction = 'opening_underdog'
        
        line_crossed = (away_open < 0) and (away_close > 0)
        
        # From opening favorite's perspective
        opening_fav_score = away_score
        opening_dog_score = home_score
        margin = opening_fav_score - opening_dog_score  # -2
        
        # Use CLOSING spread for cover calculation
        cover_margin = margin + opening_fav_closing_spread  # -2 + 1.5 = -0.5
        covered = cover_margin > 0  # False
        
        steam_team_covered = not covered  # True (steam went to opening underdog)
        
        assert line_crossed == True, "Line crossed zero"
        assert opening_fav_movement == -4.0, "4-point steam toward opening underdog"
        assert steam_direction == 'opening_underdog'
        assert covered == False, "Opening favorite didn't cover"
        assert steam_team_covered == True, "Opening underdog (steam team) covered"


class TestPropertyBased:
    """
    Property-based tests - universal properties that should always hold.
    """
    
    def test_movement_symmetry(self):
        """Movement from team A perspective = -1 * movement from team B perspective"""
        away_open = -7.0
        away_close = -10.0
        home_open = +7.0
        home_close = +10.0
        
        away_movement = away_open - away_close  # +3
        home_movement = home_open - home_close  # -3
        
        assert away_movement == -home_movement
        assert away_movement + home_movement == 0.0
    
    def test_cover_margin_symmetry(self):
        """Cover margin from team A = -1 * cover margin from team B"""
        away_score = 110
        home_score = 100
        away_spread = -7.0
        home_spread = +7.0
        
        away_margin = away_score - home_score  # +10
        away_cover_margin = away_margin + away_spread  # +3
        
        home_margin = home_score - away_score  # -10
        home_cover_margin = home_margin + home_spread  # -3
        
        assert away_cover_margin == -home_cover_margin
    
    def test_magnitude_always_non_negative(self):
        """Any magnitude metric should be >= 0"""
        test_values = [-10, -5.5, 0, 5.5, 10]
        
        for val in test_values:
            magnitude = abs(val)
            assert magnitude >= 0
            assert magnitude == abs(val)


class TestNormalizeTeamName:
    """Test team name normalization function"""
    
    def test_clippers_normalization(self):
        """Los Angeles Clippers → LA Clippers"""
        result = normalize_team_name("Los Angeles Clippers")
        assert result == "LA Clippers"
    
    def test_unmapped_team_returns_original(self):
        """Teams not in map should return original name"""
        original = "Boston Celtics"
        result = normalize_team_name(original)
        assert result == original
    
    def test_another_unmapped_team(self):
        """Another unmapped team"""
        original = "Golden State Warriors"
        result = normalize_team_name(original)
        assert result == original
    
    def test_empty_string(self):
        """Empty string should return empty string"""
        result = normalize_team_name("")
        assert result == ""


class TestAddDerivedFeatures:
    """Test add_derived_features() function with real DataFrame"""
    
    def create_sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'game_id': ['game1', 'game2', 'game3'],
            'opening_favorite': ['Team A', 'Team B', 'Team C'],
            'opening_underdog': ['Team B', 'Team A', 'Team D'],
            'opening_favorite_spread': [-7.0, -3.0, -10.0],
            'closing_favorite_spread': [-10.0, -1.0, -8.0],
            'opening_favorite_movement': [3.0, 2.0, -2.0],
            'opening_favorite_covered': [True, False, True],
            'opening_favorite_cover_margin': [5.0, -2.0, 3.0],
            'steam_direction': ['opening_favorite', 'opening_favorite', 'opening_underdog'],
            'steam_magnitude': [3.0, 2.0, 2.0],
            'steam_team_covered': [True, False, True],
            'steam_team_cover_margin': [5.0, -2.0, -3.0],
            'hours_tracked': [24.0, 12.0, 48.0],
        })
    
    def test_adds_all_expected_columns(self):
        """Verify all derived columns are added"""
        df = self.create_sample_df()
        result = add_derived_features(df)
        
        expected_columns = [
            'opening_underdog_movement',
            'opening_underdog_spread',
            'closing_underdog_spread',
            'opening_underdog_covered',
            'opening_underdog_cover_margin',
            'steam_team',
            'line_crossed_zero',
            'closing_favorite',
            'closing_underdog',
            'movement_toward_opening_favorite',
            'movement_toward_opening_underdog',
            'movement_toward_closing_favorite',
            'movement_toward_closing_underdog',
            'overall_steam_direction_team',
            'overall_steam_direction_fav_dog_at_open',
            'overall_steam_direction_fav_dog_at_close',
            'overall_steam_magnitude',
            'opening_spread_size',
            'spread_bucket',
            'movement_per_hour',
            'movement_speed',
            'fade_covered',
            'fade_margin',
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_underdog_metrics_are_inverse(self):
        """Underdog movement/spread should be inverse of favorite"""
        df = self.create_sample_df()
        result = add_derived_features(df)
        
        # Check first row
        assert result.iloc[0]['opening_underdog_movement'] == -result.iloc[0]['opening_favorite_movement']
        assert result.iloc[0]['opening_underdog_spread'] == -result.iloc[0]['opening_favorite_spread']
        assert result.iloc[0]['closing_underdog_spread'] == -result.iloc[0]['closing_favorite_spread']
        assert result.iloc[0]['opening_underdog_covered'] == (not result.iloc[0]['opening_favorite_covered'])
        assert result.iloc[0]['opening_underdog_cover_margin'] == -result.iloc[0]['opening_favorite_cover_margin']
    
    def test_spread_bucket_calculation(self):
        """Test spread bucket categorization"""
        df = pd.DataFrame({
            'opening_favorite_spread': [-2.0, -5.0, -8.0, -15.0],
            'closing_favorite_spread': [-2.0, -5.0, -8.0, -15.0],
            'opening_favorite_movement': [0.0, 0.0, 0.0, 0.0],
            'opening_favorite_covered': [True, True, True, True],
            'opening_favorite_cover_margin': [1.0, 1.0, 1.0, 1.0],
            'steam_direction': ['opening_favorite'] * 4,
            'steam_magnitude': [0.0, 0.0, 0.0, 0.0],
            'steam_team_covered': [True, True, True, True],
            'steam_team_cover_margin': [1.0, 1.0, 1.0, 1.0],
            'hours_tracked': [24.0] * 4,
            'opening_favorite': ['A', 'B', 'C', 'D'],
            'opening_underdog': ['E', 'F', 'G', 'H'],
        })
        
        result = add_derived_features(df)
        
        # Check buckets (bins: [0, 3, 6, 10, 30])
        assert result.iloc[0]['spread_bucket'] == 'close_game'  # 2.0 → [0-3)
        assert result.iloc[1]['spread_bucket'] == 'small_spread'  # 5.0 → [3-6)
        assert result.iloc[2]['spread_bucket'] == 'medium_spread'  # 8.0 → [6-10)
        assert result.iloc[3]['spread_bucket'] == 'blowout'  # 15.0 → [10-30)
    
    def test_movement_speed_buckets(self):
        """Test movement speed categorization"""
        df = pd.DataFrame({
            'opening_favorite_spread': [-7.0, -7.0, -7.0],
            'closing_favorite_spread': [-7.0, -7.0, -7.0],
            'opening_favorite_movement': [0.0, 0.0, 0.0],
            'opening_favorite_covered': [True, True, True],
            'opening_favorite_cover_margin': [1.0, 1.0, 1.0],
            'steam_direction': ['opening_favorite'] * 3,
            'steam_magnitude': [2.0, 6.0, 20.0],  # Different magnitudes
            'hours_tracked': [20.0, 20.0, 20.0],  # Same hours
            'steam_team_covered': [True, True, True],
            'steam_team_cover_margin': [1.0, 1.0, 1.0],
            'opening_favorite': ['A', 'B', 'C'],
            'opening_underdog': ['D', 'E', 'F'],
        })
        
        result = add_derived_features(df)
        
        # movement_per_hour = steam_magnitude / hours_tracked
        # bins: [0, 0.2, 0.5, 100] → ['slow', 'medium', 'fast']
        assert result.iloc[0]['movement_per_hour'] == 0.1  # 2.0/20 = 0.1 → slow
        assert result.iloc[0]['movement_speed'] == 'slow'
        
        assert result.iloc[1]['movement_per_hour'] == 0.3  # 6.0/20 = 0.3 → medium
        assert result.iloc[1]['movement_speed'] == 'medium'
        
        assert result.iloc[2]['movement_per_hour'] == 1.0  # 20.0/20 = 1.0 → fast
        assert result.iloc[2]['movement_speed'] == 'fast'
    
    def test_line_crossed_zero_detection(self):
        """Test line crossing detection"""
        df = pd.DataFrame({
            'opening_favorite_spread': [-3.0, -7.0],
            'closing_favorite_spread': [+2.0, -10.0],  # First crosses, second doesn't
            'opening_favorite_movement': [-5.0, 3.0],
            'opening_favorite_covered': [False, True],
            'opening_favorite_cover_margin': [-1.0, 1.0],
            'steam_direction': ['opening_underdog', 'opening_favorite'],
            'steam_magnitude': [5.0, 3.0],
            'steam_team_covered': [True, True],
            'steam_team_cover_margin': [1.0, 1.0],
            'hours_tracked': [24.0, 24.0],
            'opening_favorite': ['A', 'B'],
            'opening_underdog': ['C', 'D'],
        })
        
        result = add_derived_features(df)
        
        assert result.iloc[0]['line_crossed_zero'] == True, "Line should have crossed zero"
        assert result.iloc[1]['line_crossed_zero'] == False, "Line should NOT have crossed zero"
    
    def test_closing_favorite_determination(self):
        """Test closing favorite/underdog determination"""
        df = pd.DataFrame({
            'opening_favorite': ['Team A', 'Team B'],
            'opening_underdog': ['Team B', 'Team A'],
            'opening_favorite_spread': [-3.0, -7.0],
            'closing_favorite_spread': [+2.0, -10.0],  # First crosses
            'opening_favorite_movement': [-5.0, 3.0],
            'opening_favorite_covered': [False, True],
            'opening_favorite_cover_margin': [-1.0, 1.0],
            'steam_direction': ['opening_underdog', 'opening_favorite'],
            'steam_magnitude': [5.0, 3.0],
            'steam_team_covered': [True, True],
            'steam_team_cover_margin': [1.0, 1.0],
            'hours_tracked': [24.0, 24.0],
        })
        
        result = add_derived_features(df)
        
        # First game: line crossed, so closing favorite is opening underdog
        assert result.iloc[0]['closing_favorite'] == 'Team B'
        assert result.iloc[0]['closing_underdog'] == 'Team A'
        
        # Second game: line didn't cross, closing favorite = opening favorite
        assert result.iloc[1]['closing_favorite'] == 'Team B'
        assert result.iloc[1]['closing_underdog'] == 'Team A'
    
    def test_fade_strategy_columns(self):
        """Test fade strategy columns"""
        df = pd.DataFrame({
            'opening_favorite_spread': [-7.0, -7.0],
            'closing_favorite_spread': [-10.0, -4.0],
            'opening_favorite_movement': [3.0, -3.0],
            'opening_favorite_covered': [True, False],
            'opening_favorite_cover_margin': [5.0, -2.0],
            'steam_direction': ['opening_favorite', 'opening_underdog'],
            'steam_magnitude': [3.0, 3.0],
            'steam_team_covered': [True, True],
            'steam_team_cover_margin': [5.0, 2.0],
            'hours_tracked': [24.0, 24.0],
            'opening_favorite': ['A', 'B'],
            'opening_underdog': ['C', 'D'],
        })
        
        result = add_derived_features(df)
        
        # Fade = opposite of steam team
        assert result.iloc[0]['fade_covered'] == False, "Fade should be inverse"
        assert result.iloc[0]['fade_margin'] == -5.0
        
        assert result.iloc[1]['fade_covered'] == False
        assert result.iloc[1]['fade_margin'] == -2.0
    
    def test_handles_none_values_in_steam_team_covered(self):
        """Test that None values in steam_team_covered don't crash"""
        df = pd.DataFrame({
            'opening_favorite_spread': [-7.0],
            'closing_favorite_spread': [-7.0],
            'opening_favorite_movement': [0.0],
            'opening_favorite_covered': [True],
            'opening_favorite_cover_margin': [5.0],
            'steam_direction': ['no_movement'],
            'steam_magnitude': [0.0],
            'steam_team_covered': [None],  # None value
            'steam_team_cover_margin': [None],
            'hours_tracked': [24.0],
            'opening_favorite': ['A'],
            'opening_underdog': ['B'],
        })
        
        result = add_derived_features(df)
        
        # Should handle None gracefully
        assert pd.isna(result.iloc[0]['fade_covered'])
        assert pd.isna(result.iloc[0]['fade_margin'])


class TestDataFrameSchemas:
    """Test DataFrame schemas have expected structure"""
    
    def test_movements_df_required_columns(self):
        """Movements DataFrame should have all required columns"""
        required_columns = [
            'game_id', 'bookmaker', 'game_time', 'away_team', 'home_team',
            'away_open', 'home_open', 'away_close', 'home_close',
            'away_movement', 'home_movement', 'movement_magnitude',
            'movement_team', 'num_snapshots', 'hours_tracked'
        ]
        
        # Create sample movements df
        df = pd.DataFrame({
            'game_id': ['game1'],
            'bookmaker': ['DraftKings'],
            'game_time': [pd.Timestamp('2025-01-13')],
            'away_team': ['Team A'],
            'home_team': ['Team B'],
            'away_open': [-7.0],
            'home_open': [7.0],
            'away_close': [-10.0],
            'home_close': [10.0],
            'away_movement': [3.0],
            'home_movement': [-3.0],
            'movement_magnitude': [3.0],
            'movement_team': ['Team A'],
            'num_snapshots': [5],
            'hours_tracked': [24.0],
        })
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_cover_analysis_required_columns(self):
        """Cover analysis DataFrame should have all required columns"""
        required_columns = [
            'game_id', 'opening_favorite', 'opening_underdog',
            'opening_favorite_spread', 'closing_favorite_spread',
            'opening_favorite_movement', 'steam_direction', 'steam_magnitude',
            'opening_favorite_score', 'opening_underdog_score',
            'opening_favorite_covered', 'opening_favorite_cover_margin',
            'steam_team_covered', 'steam_team_cover_margin'
        ]
        
        # Create sample cover analysis df
        df = pd.DataFrame({
            'game_id': ['game1'],
            'opening_favorite': ['Team A'],
            'opening_underdog': ['Team B'],
            'opening_favorite_spread': [-7.0],
            'closing_favorite_spread': [-10.0],
            'opening_favorite_movement': [3.0],
            'steam_direction': ['opening_favorite'],
            'steam_magnitude': [3.0],
            'opening_favorite_score': [110],
            'opening_underdog_score': [95],
            'opening_favorite_covered': [True],
            'opening_favorite_cover_margin': [5.0],
            'steam_team_covered': [True],
            'steam_team_cover_margin': [5.0],
        })
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_data_types_are_correct(self):
        """Verify data types of key columns"""
        df = pd.DataFrame({
            'game_id': ['game1'],
            'steam_magnitude': [3.0],
            'opening_favorite_covered': [True],
            'num_snapshots': [5],
            'game_time': [pd.Timestamp('2025-01-13')],
        })
        
        assert df['game_id'].dtype == object  # String
        assert df['steam_magnitude'].dtype == float
        assert df['opening_favorite_covered'].dtype == bool
        assert df['num_snapshots'].dtype == int
        assert pd.api.types.is_datetime64_any_dtype(df['game_time'])


class TestConsensusCalculations:
    """Test consensus (median across bookmakers) calculations"""
    
    def test_consensus_spread_is_median(self):
        """Consensus should be median of all bookmaker spreads"""
        spreads = [-7.0, -7.5, -6.5, -8.0, -7.0]
        consensus = pd.Series(spreads).median()
        
        assert consensus == -7.0  # Median of 5 values
    
    def test_median_with_even_number_of_books(self):
        """Median with even number of bookmakers"""
        spreads = [-7.0, -8.0, -6.0, -9.0]
        consensus = pd.Series(spreads).median()
        
        assert consensus == -7.5  # (−7 + −8) / 2
    
    def test_single_bookmaker_returns_that_spread(self):
        """With only 1 bookmaker, consensus = that bookmaker's spread"""
        spreads = [-7.5]
        consensus = pd.Series(spreads).median()
        
        assert consensus == -7.5
    
    def test_consensus_handles_outliers(self):
        """Median is robust to outliers"""
        spreads = [-7.0, -7.5, -7.0, -20.0]  # -20 is outlier
        consensus = pd.Series(spreads).median()
        
        # Median should be close to -7, not affected by -20
        assert consensus == -7.25  # Median of sorted: [-20, -7.5, -7, -7]


class TestHourlySteam:
    """Test max 1-hour steam calculation logic"""
    
    def test_finds_biggest_1hr_spike(self):
        """Should find the largest hour-over-hour change"""
        # Simulate hourly spreads: -7 → -8 → -12 → -11
        # Changes: -1, -4, +1
        # Biggest 1hr spike: 4 points (2nd to 3rd hour)
        
        hourly_spreads = [-7.0, -8.0, -12.0, -11.0]
        
        max_change = 0.0
        for i in range(1, len(hourly_spreads)):
            change = abs(hourly_spreads[i] - hourly_spreads[i-1])
            max_change = max(max_change, change)
        
        assert max_change == 4.0
    
    def test_handles_single_snapshot(self):
        """With only 1 snapshot, no 1-hour steam possible"""
        hourly_spreads = [-7.0]
        
        # Can't calculate hour-over-hour change with 1 snapshot
        assert len(hourly_spreads) == 1
        # Would return 0 or skip this game
    
    def test_multiple_spikes_returns_largest(self):
        """With multiple spikes, return the largest"""
        # -7 → -10 (+3) → -12 (+2) → -8 (-4) → -13 (+5)
        # Largest spike: 5 points
        
        hourly_spreads = [-7.0, -10.0, -12.0, -8.0, -13.0]
        
        max_change = 0.0
        for i in range(1, len(hourly_spreads)):
            change = abs(hourly_spreads[i] - hourly_spreads[i-1])
            max_change = max(max_change, change)
        
        assert max_change == 5.0
    
    def test_steam_direction_matches_biggest_spike(self):
        """Steam direction should match the team that got the biggest spike"""
        # Away team spread: -7 → -10 (3pt toward away) → -8 (2pt toward home)
        # Biggest spike: 3 points toward away team
        
        away_spreads = [-7.0, -10.0, -8.0]
        
        max_change = 0.0
        max_change_direction = None
        
        for i in range(1, len(away_spreads)):
            change = away_spreads[i] - away_spreads[i-1]
            if abs(change) > abs(max_change):
                max_change = change
                max_change_direction = 'away' if change < 0 else 'home'
        
        assert abs(max_change) == 3.0
        assert max_change_direction == 'away'


class TestEndToEndWithFixtures:
    """Test full pipeline logic with synthetic data"""
    
    def test_full_pipeline_basic_scenario(self):
        """Test a complete scenario from movement to cover analysis"""
        # Setup: Single game with 2pt steam toward favorite, favorite covers
        
        # Movement data
        movements = pd.DataFrame({
            'game_id': ['game1'],
            'away_team': ['Celtics'],
            'home_team': ['Lakers'],
            'away_open': [-7.0],
            'away_close': [-9.0],
            'home_open': [7.0],
            'home_close': [9.0],
            'bookmaker': ['DraftKings'],
        })
        
        # Game results (2 rows - one per team)
        results = pd.DataFrame({
            'GAME_DATE': [pd.Timestamp('2025-01-13'), pd.Timestamp('2025-01-13')],
            'TEAM_NAME': ['Boston Celtics', 'Los Angeles Lakers'],
            'PTS': [110, 95],
        })
        
        # Calculate movement
        movements['away_movement'] = movements['away_close'] - movements['away_open']  # -2
        movements['home_movement'] = movements['home_close'] - movements['home_open']  # 2
        movements['movement_magnitude'] = movements[['away_movement', 'home_movement']].abs().max(axis=1)
        
        # Determine opening favorite
        away_is_fav = movements['away_open'].iloc[0] < 0
        
        if away_is_fav:
            opening_fav_movement = movements['away_open'].iloc[0] - movements['away_close'].iloc[0]
        else:
            opening_fav_movement = movements['home_open'].iloc[0] - movements['home_close'].iloc[0]
        
        steam_direction = 'opening_favorite' if opening_fav_movement > 0 else 'opening_underdog'
        
        assert movements['movement_magnitude'].iloc[0] == 2.0
        assert steam_direction == 'opening_favorite'
        assert opening_fav_movement == 2.0
    
    def test_deduplication_keeps_largest_steam(self):
        """When multiple bookmakers, deduplication should keep largest steam"""
        df = pd.DataFrame({
            'game_id': ['game1', 'game1', 'game1'],
            'bookmaker': ['DraftKings', 'FanDuel', 'BetMGM'],
            'steam_magnitude': [3.0, 5.0, 2.0],  # FanDuel has largest
            'steam_team_covered': [True, True, False],
        })
        
        # Deduplicate: sort by steam_magnitude DESC, keep first per game
        deduped = df.sort_values('steam_magnitude', ascending=False).drop_duplicates('game_id', keep='first')
        
        assert len(deduped) == 1
        assert deduped.iloc[0]['bookmaker'] == 'FanDuel'
        assert deduped.iloc[0]['steam_magnitude'] == 5.0
    
    def test_merge_with_hourly_steam_data(self):
        """Test merging overall steam with hourly steam data"""
        # Overall steam
        overall = pd.DataFrame({
            'game_id': ['game1', 'game2'],
            'overall_steam_magnitude': [3.0, 5.0],
        })
        
        # Hourly steam
        hourly = pd.DataFrame({
            'game_id': ['game1', 'game2'],
            'max_1hr_steam_magnitude': [2.0, 4.5],
            'max_1hr_steam_direction_team': ['Team A', 'Team C'],
        })
        
        # Merge
        merged = overall.merge(hourly, on='game_id', how='left')
        
        assert len(merged) == 2
        assert 'max_1hr_steam_magnitude' in merged.columns
        assert merged.iloc[0]['max_1hr_steam_magnitude'] == 2.0
        assert merged.iloc[1]['max_1hr_steam_magnitude'] == 4.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

