# app/services/prediction_service.py

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import poisson

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for dynamic weighting and confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for making a recommendation
REQUIRED_CONSISTENCY = 0.8       # 80% consistency threshold

# ===========================
# Utility Functions
# ===========================

def aggregate_weighted_outcomes(outcomes: List[Tuple[str, float]]) -> Counter:
    """Aggregate weighted outcomes."""
    result_counter = Counter()
    for outcome, weight in outcomes:
        result_counter[outcome] += weight
    return result_counter

def dynamic_weight_adjustment(outcomes: List[Tuple[str, float]], data_quality_factor: float) -> List[Tuple[str, float]]:
    """Dynamically adjust the weights based on data quality."""
    adjusted_outcomes = []
    for outcome, weight in outcomes:
        adjusted_weight = weight * data_quality_factor
        adjusted_outcomes.append((outcome, adjusted_weight))
    return adjusted_outcomes

def log_insights(event_data, predictions, confidence_score):
    """Log important insights for model improvement."""
    home_team = event_data['home_team']['name']
    away_team = event_data['away_team']['name']

    logger.info(f"Prediction for {home_team} vs {away_team}: {predictions}")
    logger.info(f"Confidence score: {confidence_score:.2%}")

def is_h2h_recent_and_sufficient(h2h_data: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Check if H2H data is recent and sufficient."""
    recent_matches = []
    current_year = datetime.now().year
    for match in h2h_data:
        try:
            match_year = datetime.fromisoformat(match['fixture']['date'].replace('Z', '+00:00')).year
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing match date: {e}")
            continue
        if match_year >= current_year - 2:
            recent_matches.append(match)
    return len(recent_matches) >= 2, recent_matches

def filter_valid_matches(matches: List[Dict]) -> List[Dict]:
    """Filter out None or invalid matches from the list."""
    return [match for match in matches if isinstance(match, dict)]

def safe_get_numeric(dictionary: Dict, keys: List[str], default=0) -> float:
    """Safely get a numeric value from a nested dictionary."""
    value = dictionary
    for key in keys:
        value = value.get(key)
        if value is None:
            return default
    return float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit() else default

# ===========================
# Data Processing Functions
# ===========================

def calculate_dynamic_threshold(matches: List[Dict]) -> float:
    """Calculate the dynamic threshold for goals based on match data."""
    total_goals = sum(
        safe_get_numeric(match, ['goals', 'home']) + safe_get_numeric(match, ['goals', 'away'])
        for match in matches
    )
    total_matches = len(matches)
    average_goals_per_match = total_goals / total_matches if total_matches > 0 else 0.0
    return average_goals_per_match

def round_dynamic_threshold(value: float) -> float:
    """Round dynamic threshold values to the nearest standard over/under goal thresholds from 1.5 to 5.5."""
    thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
    for threshold in thresholds:
        if value <= threshold:
            return threshold
    return 5.5  # For values greater than 5.5

def analyze_recent_performance(last_matches: List[Dict]) -> Tuple[List[Tuple[str, float]], float]:
    """Analyze recent performance with exponential decay weights."""
    outcomes = []
    total_goals = 0.0  # Ensure default to zero
    now = datetime.now(timezone.utc)

    for match in last_matches:
        try:
            if not isinstance(match, dict):
                continue  # Skip if match is not a dict
            fixture = match.get('fixture', {})
            if not fixture:
                continue
            match_date_str = fixture.get('date')
            if not match_date_str:
                continue
            match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
            days_ago = (now - match_date).days
            decay_factor = 0.5 ** (days_ago / 180)  # Half-life of 180 days
            weight = decay_factor

            home_goals = safe_get_numeric(match, ['goals', 'home'])
            away_goals = safe_get_numeric(match, ['goals', 'away'])
            total_goals += home_goals + away_goals

            if home_goals > away_goals:
                outcomes.append(("Home Win", weight))
            elif away_goals > home_goals:
                outcomes.append(("Away Win", weight))
            else:
                outcomes.append(("Draw", weight))
        except Exception as e:
            logger.error(f"Error analyzing recent performance: {e}")
            continue

    return outcomes, total_goals

def detect_momentum(last_matches: List[Dict]) -> Dict[str, int]:
    """Detect win/lose/draw streaks from the last matches."""
    win_streak = 0
    lose_streak = 0
    draw_streak = 0

    for match in last_matches:
        home_goals = safe_get_numeric(match, ['goals', 'home'])
        away_goals = safe_get_numeric(match, ['goals', 'away'])

        if home_goals > away_goals:
            win_streak += 1
            lose_streak = 0
            draw_streak = 0
        elif away_goals > home_goals:
            lose_streak += 1
            win_streak = 0
            draw_streak = 0
        else:
            draw_streak += 1
            win_streak = 0
            lose_streak = 0

    return {
        "win_streak": win_streak,
        "lose_streak": lose_streak,
        "draw_streak": draw_streak
    }

def poisson_prediction(avg_goals_home: float, avg_goals_away: float) -> Dict[str, float]:
    """Use Poisson distribution to predict match outcome probabilities."""
    max_goals = 5
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [avg_goals_home, avg_goals_away]]

    home_win_prob = 0.0  # Initialize probabilities
    away_win_prob = 0.0
    draw_prob = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = team_pred[0][i] * team_pred[1][j]
            if i > j:
                home_win_prob += prob
            elif i < j:
                away_win_prob += prob
            else:
                draw_prob += prob

    total_prob = home_win_prob + away_win_prob + draw_prob
    if total_prob == 0:
        return {"Home Win": 0.0, "Away Win": 0.0, "Draw": 0.0}
    return {
        "Home Win": home_win_prob / total_prob,
        "Away Win": away_win_prob / total_prob,
        "Draw": draw_prob / total_prob
    }

def calculate_correct_score_probs(avg_home_goals, avg_away_goals):
    """Calculate probabilities for correct score predictions."""
    max_goals = 5
    score_probs = {}
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson.pmf(home_goals, avg_home_goals) * poisson.pmf(away_goals, avg_away_goals)
            score_probs[f"{home_goals}-{away_goals}"] = prob
    return score_probs

def analyze_goals_per_half(last_matches: List[Dict]) -> Tuple[float, float]:
    """Calculate average goals per half for the team."""
    first_half_goals = []
    second_half_goals = []

    for match in last_matches:
        try:
            home_ht_goals = safe_get_numeric(match, ['score', 'halftime', 'home'])
            away_ht_goals = safe_get_numeric(match, ['score', 'halftime', 'away'])
            total_first_half_goals = home_ht_goals + away_ht_goals
            first_half_goals.append(total_first_half_goals)

            full_time_home_goals = safe_get_numeric(match, ['goals', 'home'])
            full_time_away_goals = safe_get_numeric(match, ['goals', 'away'])
            total_full_time_goals = full_time_home_goals + full_time_away_goals

            second_half_goals.append(total_full_time_goals - total_first_half_goals)
        except Exception as e:
            logger.error(f"Error analyzing goals per half: {e}")
            continue

    avg_first_half_goals = np.mean(first_half_goals) if first_half_goals else 0.0
    avg_second_half_goals = np.mean(second_half_goals) if second_half_goals else 0.0

    return avg_first_half_goals, avg_second_half_goals

def calculate_half_time_draw_probability(avg_home_first_half_goals, avg_away_first_half_goals):
    """Calculate the probability of a half-time draw using Poisson distribution."""
    max_goals = 3
    home_team_pred = [poisson.pmf(i, avg_home_first_half_goals) for i in range(max_goals + 1)]
    away_team_pred = [poisson.pmf(i, avg_away_first_half_goals) for i in range(max_goals + 1)]

    draw_prob = 0.0

    for i in range(max_goals + 1):
        prob = home_team_pred[i] * away_team_pred[i]
        draw_prob += prob

    return draw_prob

def predict_highest_scoring_half(avg_first_half_goals, avg_second_half_goals):
    """Predict which half will have more goals."""
    if avg_first_half_goals > avg_second_half_goals:
        return "First Half"
    elif avg_second_half_goals > avg_first_half_goals:
        return "Second Half"
    else:
        return "Equal"

def calculate_total_goals_conceded(matches: List[Dict], team_side: str) -> int:
    """Calculate total goals conceded by the team."""
    total_conceded = 0
    for match in matches:
        try:
            if team_side == 'home':
                conceded = safe_get_numeric(match, ['goals', 'away'])
            elif team_side == 'away':
                conceded = safe_get_numeric(match, ['goals', 'home'])
            else:
                conceded = 0
            total_conceded += conceded
        except Exception as e:
            logger.error(f"Error calculating goals conceded: {e}")
            continue
    return total_conceded

# ===========================
# calculate_confidence Function
# ===========================

def calculate_confidence(home_outcomes: List[Tuple[str, float]],
                        away_outcomes: List[Tuple[str, float]],
                        h2h_outcomes: List[Tuple[str, float]]) -> float:
    """
    Calculate a confidence score based on outcome consistency.
    """
    total_outcomes = home_outcomes + away_outcomes + h2h_outcomes
    outcome_counts = aggregate_weighted_outcomes(total_outcomes)

    home_wins = outcome_counts.get("Home Win", 0)
    away_wins = outcome_counts.get("Away Win", 0)
    draws = outcome_counts.get("Draw", 0)

    total_weights = sum(outcome_counts.values())

    if total_weights == 0:
        logger.warning("Total weights are zero. Returning confidence score of 0.0.")
        return 0.0

    max_outcome_weight = max(home_wins, away_wins, draws)
    confidence = max_outcome_weight / total_weights

    return confidence

# ===========================
# Prediction Function
# ===========================

def predict_football(event_data, custom_weights=None):
    """Predict outcomes for a football match."""
    custom_weights = custom_weights or {
        'recent_weight': 3,
        'mid_recent_weight': 2,
        'older_weight': 1,
        'h2h_weight': 1.5,
        'first_half_draw_weight': 1
    }

    home_team = event_data['home_team']['name']
    away_team = event_data['away_team']['name']

    match_status = event_data.get('fixture', {}).get('status', {}).get('short', '')

    if match_status == 'PST':
        logger.info(f"Match {home_team} vs {away_team} is postponed.")
        return {
            "event": event_data,
            "final_predictions": {
                "predictions": {},
                "best_bet": "Match postponed, prediction not available."
            }
        }

    h2h_data = event_data.get('h2h', [])
    if isinstance(h2h_data, dict):
        h2h_data = h2h_data.get('response', [])

    # Ensure last_10_home and last_10_away are lists and not None
    last_10_home = event_data.get('last_10_home') or []
    last_10_away = event_data.get('last_10_away') or []

    # Filter out invalid matches
    h2h_data = filter_valid_matches(h2h_data)
    last_10_home = filter_valid_matches(last_10_home)
    last_10_away = filter_valid_matches(last_10_away)

    # No prediction if both H2H and recent data are insufficient
    if len(h2h_data) < 2 and len(last_10_home) < 3 and len(last_10_away) < 3:
        logger.warning(f"Insufficient data for {home_team} vs {away_team}.")
        return {
            "event": event_data,
            "final_predictions": {
                "predictions": {},
                "best_bet": "No prediction available due to insufficient data."
            }
        }

    data_quality_factor = min(1.0, (len(last_10_home) + len(last_10_away)) / 20.0)
    logger.info(f"Data quality factor for {home_team} vs {away_team}: {data_quality_factor:.2f}")

    h2h_sufficient, recent_h2h_data = is_h2h_recent_and_sufficient(h2h_data)
    h2h_to_use = recent_h2h_data if h2h_sufficient else []

    outcomes = []
    goal_counts = []

    for match in h2h_to_use:
        if not isinstance(match, dict):
            continue
        home_goals = safe_get_numeric(match, ['goals', 'home'])
        away_goals = safe_get_numeric(match, ['goals', 'away'])
        goal_counts.append((home_goals, away_goals))

        if home_goals > away_goals:
            outcomes.append(("Home Win", custom_weights['h2h_weight']))
        elif away_goals > home_goals:
            outcomes.append(("Away Win", custom_weights['h2h_weight']))
        else:
            outcomes.append(("Draw", custom_weights['h2h_weight']))

    # Analyze recent performance
    home_recent_outcomes, home_recent_goals = analyze_recent_performance(last_10_home)
    away_recent_outcomes, away_recent_goals = analyze_recent_performance(last_10_away)

    # Ensure home_recent_goals and away_recent_goals are not None
    home_recent_goals = home_recent_goals if home_recent_goals is not None else 0.0
    away_recent_goals = away_recent_goals if away_recent_goals is not None else 0.0

    home_recent_outcomes = dynamic_weight_adjustment(home_recent_outcomes, data_quality_factor)
    away_recent_outcomes = dynamic_weight_adjustment(away_recent_outcomes, data_quality_factor)

    # Detect momentum
    home_momentum = detect_momentum(last_10_home[:5])
    away_momentum = detect_momentum(last_10_away[:5])

    # Factoring momentum
    if home_momentum['win_streak'] >= 3:
        logger.info(f"{home_team} is on a winning streak.")
        outcomes.append(("Home Win", 1))
    if away_momentum['win_streak'] >= 3:
        logger.info(f"{away_team} is on a winning streak.")
        outcomes.append(("Away Win", 1))

    if home_momentum['lose_streak'] >= 3:
        logger.info(f"{home_team} is on a losing streak.")
        outcomes.append(("Away Win", 1))
    if away_momentum['lose_streak'] >= 3:
        logger.info(f"{away_team} is on a losing streak.")
        outcomes.append(("Home Win", 1))

    combined_outcomes = outcomes + home_recent_outcomes + away_recent_outcomes
    outcome_counts = aggregate_weighted_outcomes(combined_outcomes)

    # Compute probabilities
    total_weight = sum(outcome_counts.values())

    home_win_weight = outcome_counts.get("Home Win", 0)
    away_win_weight = outcome_counts.get("Away Win", 0)
    draw_weight = outcome_counts.get("Draw", 0)

    home_win_prob = home_win_weight / total_weight if total_weight > 0 else 0.0
    away_win_prob = away_win_weight / total_weight if total_weight > 0 else 0.0
    draw_prob = draw_weight / total_weight if total_weight > 0 else 0.0

    # Incorporate Poisson distribution
    avg_home_goals = home_recent_goals / len(last_10_home) if len(last_10_home) >= 10 else 0.0
    avg_away_goals = away_recent_goals / len(last_10_away) if len(last_10_away) >= 10 else 0.0

    poisson_probs = poisson_prediction(avg_home_goals, avg_away_goals)

    # Adjust probabilities
    home_win_prob = (home_win_prob + poisson_probs["Home Win"]) / 2
    away_win_prob = (away_win_prob + poisson_probs["Away Win"]) / 2
    draw_prob = (draw_prob + poisson_probs["Draw"]) / 2

    # Adjusted prediction logic
    predictions = {}

    if home_win_prob + draw_prob >= 0.6:
        predictions["match_outcome"] = "Home Win or Draw"
    elif away_win_prob + draw_prob >= 0.6:
        predictions["match_outcome"] = "Away Win or Draw"
    elif draw_prob >= 0.6:
        predictions["match_outcome"] = "Draw"
    else:
        predictions["match_outcome"] = "No clear prediction"

    # Dynamic threshold calculation
    dynamic_threshold_raw = calculate_dynamic_threshold(h2h_to_use if h2h_sufficient else last_10_home + last_10_away)
    dynamic_threshold = round_dynamic_threshold(dynamic_threshold_raw)

    # Ensure home_recent_goals and away_recent_goals are numbers
    if not isinstance(home_recent_goals, (int, float)):
        home_recent_goals = 0.0
    if not isinstance(away_recent_goals, (int, float)):
        away_recent_goals = 0.0

    total_goals_scored = (
        sum((home or 0) + (away or 0) for home, away in goal_counts) + home_recent_goals + away_recent_goals
    )
    total_matches_counted = len(h2h_to_use) + len(last_10_home) + len(last_10_away)
    average_goals_per_match = total_goals_scored / total_matches_counted if total_matches_counted > 0 else 0.0

    # Average goals for home and away teams
    average_home_goals = avg_home_goals
    average_away_goals = avg_away_goals

    both_teams_score = any((home or 0) > 0 and (away or 0) > 0 for home, away in goal_counts)

    # Monte Carlo simulation using Poisson distribution
    def refined_monte_carlo_simulation(avg_home_goals, avg_away_goals, num_simulations=10000):
        """Refined Monte Carlo simulation using Poisson distribution."""
        home_win = 0
        away_win = 0
        draw = 0

        for _ in range(num_simulations):
            home_goals_sim = np.random.poisson(avg_home_goals)
            away_goals_sim = np.random.poisson(avg_away_goals)

            if home_goals_sim > away_goals_sim:
                home_win += 1
            elif away_goals_sim > home_goals_sim:
                away_win += 1
            else:
                draw += 1

        total = home_win + away_win + draw
        if total == 0:
            return {"Home Win": 0.0, "Away Win": 0.0, "Draw": 0.0}
        return {
            "Home Win": home_win / total,
            "Away Win": away_win / total,
            "Draw": draw / total
        }

    simulation_results = refined_monte_carlo_simulation(avg_home_goals, avg_away_goals)

    # Correct score probabilities
    score_probs = calculate_correct_score_probs(avg_home_goals, avg_away_goals)
    most_likely_score = max(score_probs, key=score_probs.get, default="0-0")
    predictions["most_likely_score"] = most_likely_score

    # Goals per half analysis
    avg_match_first_half_goals, avg_match_second_half_goals = analyze_goals_per_half(last_10_home + last_10_away)
    avg_home_first_half_goals, avg_home_second_half_goals = analyze_goals_per_half(last_10_home)
    avg_away_first_half_goals, avg_away_second_half_goals = analyze_goals_per_half(last_10_away)

    # Highest Scoring Half Prediction
    predictions["highest_scoring_half"] = predict_highest_scoring_half(avg_match_first_half_goals, avg_match_second_half_goals)

    # Half-Time Draw Probability
    half_time_draw_prob = calculate_half_time_draw_probability(avg_home_first_half_goals, avg_away_first_half_goals)
    if half_time_draw_prob >= 0.5:
        predictions["half_time_draw"] = "Half Time Draw"
    else:
        predictions["half_time_draw"] = "Half Time Not Draw"

    # Over/Under Goals in First Half for Home Team
    home_first_half_goals_threshold = round_dynamic_threshold(avg_home_first_half_goals)
    if avg_home_first_half_goals >= home_first_half_goals_threshold:
        predictions["home_first_half_over"] = f"Home Over ({home_first_half_goals_threshold}) Goals in First Half"
    else:
        predictions["home_first_half_under"] = f"Home Under ({home_first_half_goals_threshold}) Goals in First Half"

    # Over/Under Goals in Second Half for Home Team
    home_second_half_goals_threshold = round_dynamic_threshold(avg_home_second_half_goals)
    if avg_home_second_half_goals >= home_second_half_goals_threshold:
        predictions["home_second_half_over"] = f"Home Over ({home_second_half_goals_threshold}) Goals in Second Half"
    else:
        predictions["home_second_half_under"] = f"Home Under ({home_second_half_goals_threshold}) Goals in Second Half"

    # Over/Under Goals in First Half for Away Team
    away_first_half_goals_threshold = round_dynamic_threshold(avg_away_first_half_goals)
    if avg_away_first_half_goals >= away_first_half_goals_threshold:
        predictions["away_first_half_over"] = f"Away Over ({away_first_half_goals_threshold}) Goals in First Half"
    else:
        predictions["away_first_half_under"] = f"Away Under ({away_first_half_goals_threshold}) Goals in First Half"

    # Over/Under Goals in Second Half for Away Team
    away_second_half_goals_threshold = round_dynamic_threshold(avg_away_second_half_goals)
    if avg_away_second_half_goals >= away_second_half_goals_threshold:
        predictions["away_second_half_over"] = f"Away Over ({away_second_half_goals_threshold}) Goals in Second Half"
    else:
        predictions["away_second_half_under"] = f"Away Under ({away_second_half_goals_threshold}) Goals in Second Half"

    # Total Under X Goals Prediction
    predictions["total_under_x_goals"] = (
        f"Total Under ({dynamic_threshold:.1f}) Goals"
        if average_goals_per_match <= dynamic_threshold
        else f"Total Over ({dynamic_threshold:.1f}) Goals"
    )

    # Adjusted predictions based on averages and trends
    predictions["total_over_x_goals"] = (
        f"Total Over ({dynamic_threshold:.1f}) Goals"
        if average_goals_per_match >= dynamic_threshold
        else f"Total Under ({dynamic_threshold:.1f}) Goals"
    )

    # Both Teams to Score Prediction
    if both_teams_score or (home_win_prob > 0.4 and away_win_prob > 0.4):
        predictions["both_teams_to_score"] = "Both Teams to Score"
    else:
        predictions["both_teams_to_score"] = "Both Teams Not to Score"

    # Confidence Score
    confidence_score = calculate_confidence(
        home_recent_outcomes, away_recent_outcomes, outcomes
    )
    predictions["confidence_score"] = f"{confidence_score:.2%}"

    # Monte Carlo Simulation Results
    predictions["monte_carlo_simulation"] = simulation_results

    # ===========================
    # Improved Best Bet Logic with Consistency Checks
    # ===========================

    best_bet_options = []

    # Function to calculate team success rate
    def calculate_team_success_rate(last_matches):
        """Calculate the win rate of a team in the last matches."""
        wins = 0
        total_matches = len(last_matches)
        for match in last_matches:
            home_goals = safe_get_numeric(match, ['goals', 'home'])
            away_goals = safe_get_numeric(match, ['goals', 'away'])
            # Assuming 'home_team' is the team in question
            if home_goals > away_goals:
                wins += 1
        success_rate = (wins / total_matches) if total_matches > 0 else 0
        logger.debug(f"Team success rate: {success_rate:.2%}, Wins: {wins}, Total Matches: {total_matches}")
        return success_rate, wins

    # Function to check goals consistency
    def check_goals_consistency(last_matches, threshold, over=True):
        """Check if the teams consistently score over or under the threshold."""
        consistent_matches = 0
        for match in last_matches:
            home_goals = safe_get_numeric(match, ['goals', 'home'])
            away_goals = safe_get_numeric(match, ['goals', 'away'])
            total_goals = home_goals + away_goals
            if over and total_goals > threshold:
                consistent_matches += 1
            elif not over and total_goals < threshold:
                consistent_matches += 1
        total_matches = len(last_matches)
        consistency_rate = consistent_matches / total_matches if total_matches else 0
        logger.debug(f"Goals consistency rate: {consistency_rate:.2%} (Over {threshold} Goals: {over}), Consistent Matches: {consistent_matches}, Total Matches: {total_matches}")
        return consistency_rate >= REQUIRED_CONSISTENCY

    # Function to check half-time draw consistency
    def check_half_time_draw_consistency(last_matches):
        """Check if the teams have a high rate of half-time draws."""
        half_time_draws = 0
        for match in last_matches:
            home_ht_goals = safe_get_numeric(match, ['score', 'halftime', 'home'])
            away_ht_goals = safe_get_numeric(match, ['score', 'halftime', 'away'])
            if home_ht_goals == away_ht_goals:
                half_time_draws += 1
        total_matches = len(last_matches)
        draw_rate = (half_time_draws / total_matches) if total_matches > 0 else 0
        logger.debug(f"Half-time draw rate: {draw_rate:.2%}, Half-time Draws: {half_time_draws}, Total Matches: {total_matches}")
        return draw_rate >= 0.7  # Adjusted to 70% half-time draws

    # Function to check both teams to score consistency
    def check_both_teams_to_score_consistency(last_matches):
        """Check if both teams have a high rate of both teams scoring."""
        both_teams_score_matches = 0
        for match in last_matches:
            home_goals = safe_get_numeric(match, ['goals', 'home'])
            away_goals = safe_get_numeric(match, ['goals', 'away'])
            if home_goals > 0 and away_goals > 0:
                both_teams_score_matches += 1
        total_matches = len(last_matches)
        consistency_rate = both_teams_score_matches / total_matches if total_matches else 0
        logger.debug(f"Both teams to score rate: {consistency_rate:.2%}, Matches with both teams scoring: {both_teams_score_matches}, Total Matches: {total_matches}")
        return consistency_rate >= 0.7  # Adjusted to 70% of matches

    # Function to check highest scoring half consistency
    def check_highest_scoring_half_consistency(last_matches, predicted_half):
        """Check if there's consistency in which half is higher scoring."""
        consistent_matches = 0
        for match in last_matches:
            ht_goals = safe_get_numeric(match, ['score', 'halftime', 'home']) + safe_get_numeric(match, ['score', 'halftime', 'away'])
            ft_goals = safe_get_numeric(match, ['goals', 'home']) + safe_get_numeric(match, ['goals', 'away'])
            sh_goals = ft_goals - ht_goals
            if predicted_half == "First Half" and ht_goals > sh_goals:
                consistent_matches += 1
            elif predicted_half == "Second Half" and sh_goals > ht_goals:
                consistent_matches += 1
            elif predicted_half == "Equal" and ht_goals == sh_goals:
                consistent_matches += 1
        total_matches = len(last_matches)
        consistency_rate = consistent_matches / total_matches if total_matches else 0
        logger.debug(f"Highest scoring half consistency rate: {consistency_rate:.2%}, Consistent Matches: {consistent_matches}, Total Matches: {total_matches}")
        return consistency_rate >= 0.7  # Adjusted to 70% consistency

    # Function to check team half goals consistency
    def check_team_half_goals_consistency(last_matches, team_side, half, threshold, over=True, required_consistency=REQUIRED_CONSISTENCY):
        """Check if the team consistently scores over or under the threshold in the specified half."""
        consistent_matches = 0
        for match in last_matches:
            goals_key = 'home' if team_side == 'home' else 'away'

            if half == 'first':
                half_goals = safe_get_numeric(match, ['score', 'halftime', goals_key])
            elif half == 'second':
                full_goals = safe_get_numeric(match, ['goals', goals_key])
                first_half_goals = safe_get_numeric(match, ['score', 'halftime', goals_key])
                half_goals = full_goals - first_half_goals
            else:
                continue

            if over and half_goals > threshold:
                consistent_matches += 1
            elif not over and half_goals < threshold:
                consistent_matches += 1

        total_matches = len(last_matches)
        consistency_rate = consistent_matches / total_matches if total_matches else 0
        logger.debug(f"Team {team_side} {half} half goals consistency rate: {consistency_rate:.2%} (Over {threshold} Goals: {over}), Consistent Matches: {consistent_matches}, Total Matches: {total_matches}")
        return consistency_rate >= required_consistency  # Set to 80% consistency

    # Function to calculate total goals conceded by a team
    def calculate_total_goals_conceded(last_matches, team_side):
        """Calculate total goals conceded by the team."""
        total_conceded = 0
        for match in last_matches:
            if team_side == 'home':
                conceded = safe_get_numeric(match, ['goals', 'away'])
            elif team_side == 'away':
                conceded = safe_get_numeric(match, ['goals', 'home'])
            else:
                conceded = 0
            total_conceded += conceded
        return total_conceded

    # Function to check team consistencies
    def calculate_team_consistencies(last_matches_home, last_matches_away):
        """Calculate team streaks and goals conceded."""
        home_win_streak = home_momentum.get('win_streak', 0)
        away_win_streak = away_momentum.get('win_streak', 0)
        total_goals_conceded_home = calculate_total_goals_conceded(last_matches_home, 'home')
        total_goals_conceded_away = calculate_total_goals_conceded(last_matches_away, 'away')
        return home_win_streak, away_win_streak, total_goals_conceded_home, total_goals_conceded_away

    # Calculate team consistencies
    home_win_streak, away_win_streak, total_goals_conceded_home, total_goals_conceded_away = calculate_team_consistencies(last_10_home, last_10_away)

    # Match Outcome Predictions with Conditions
    if predictions["match_outcome"] == "Home Win or Draw":
        home_success_rate, home_wins = calculate_team_success_rate(last_10_home)
        if (home_success_rate >= REQUIRED_CONSISTENCY and home_wins >= 4):
            # Assign a high score based on confidence
            best_bet_options.append(("Home Win or Draw", confidence_score))
    elif predictions["match_outcome"] == "Away Win or Draw":
        away_success_rate, away_wins = calculate_team_success_rate(last_10_away)
        if (away_success_rate >= REQUIRED_CONSISTENCY and away_wins >= 4):
            best_bet_options.append(("Away Win or Draw", confidence_score))

    # Half-Time Draw Prediction with Consistency Check
    if predictions.get("half_time_draw") == "Half Time Draw":
        if check_half_time_draw_consistency(h2h_to_use + last_10_home + last_10_away):
            best_bet_options.append(("Half Time Draw", confidence_score))

    # Total Goals Predictions with Consistency Check
    # For Total Under Goals Prediction
    if predictions["total_under_x_goals"].startswith("Total Under"):
        threshold = float(predictions["total_under_x_goals"].split("(")[1].split(")")[0])
        if check_goals_consistency(last_10_home + last_10_away, threshold, over=False):
            best_bet_options.append((predictions["total_under_x_goals"], confidence_score))

    # For Total Over Goals Prediction
    if predictions["total_over_x_goals"].startswith("Total Over"):
        threshold = float(predictions["total_over_x_goals"].split("(")[1].split(")")[0])
        if check_goals_consistency(last_10_home + last_10_away, threshold, over=True):
            best_bet_options.append((predictions["total_over_x_goals"], confidence_score))

    # Both Teams to Score Prediction with Consistency Check
    if predictions.get("both_teams_to_score") == "Both Teams to Score":
        if check_both_teams_to_score_consistency(h2h_to_use + last_10_home + last_10_away):
            best_bet_options.append(("Both Teams to Score", confidence_score))

    # Highest Scoring Half Prediction with Consistency Check
    if predictions.get("highest_scoring_half") in ["First Half", "Second Half", "Equal"]:
        if check_highest_scoring_half_consistency(h2h_to_use + last_10_home + last_10_away, predictions.get("highest_scoring_half")):
            if predictions.get("highest_scoring_half") == "Equal":
                best_bet_options.append(("Both Halves Equal Scoring", confidence_score))
            else:
                best_bet_options.append((f"{predictions['highest_scoring_half']} Higher Scoring", confidence_score))

    # Over/Under Goals Predictions for Home Team with Consistency Check
    # First Half
    if "home_first_half_over" in predictions:
        threshold = float(predictions["home_first_half_over"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_home, 'home', 'first', threshold, over=True):
            best_bet_options.append((predictions["home_first_half_over"], confidence_score))
    elif "home_first_half_under" in predictions:
        threshold = float(predictions["home_first_half_under"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_home, 'home', 'first', threshold, over=False):
            best_bet_options.append((predictions["home_first_half_under"], confidence_score))

    # Second Half
    if "home_second_half_over" in predictions:
        threshold = float(predictions["home_second_half_over"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_home, 'home', 'second', threshold, over=True):
            best_bet_options.append((predictions["home_second_half_over"], confidence_score))
    elif "home_second_half_under" in predictions:
        threshold = float(predictions["home_second_half_under"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_home, 'home', 'second', threshold, over=False):
            best_bet_options.append((predictions["home_second_half_under"], confidence_score))

    # Over/Under Goals Predictions for Away Team with Consistency Check
    # First Half
    if "away_first_half_over" in predictions:
        threshold = float(predictions["away_first_half_over"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_away, 'away', 'first', threshold, over=True):
            best_bet_options.append((predictions["away_first_half_over"], confidence_score))
    elif "away_first_half_under" in predictions:
        threshold = float(predictions["away_first_half_under"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_away, 'away', 'first', threshold, over=False):
            best_bet_options.append((predictions["away_first_half_under"], confidence_score))

    # Second Half
    if "away_second_half_over" in predictions:
        threshold = float(predictions["away_second_half_over"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_away, 'away', 'second', threshold, over=True):
            best_bet_options.append((predictions["away_second_half_over"], confidence_score))
    elif "away_second_half_under" in predictions:
        threshold = float(predictions["away_second_half_under"].split("(")[1].split(")")[0])
        if check_team_half_goals_consistency(last_10_away, 'away', 'second', threshold, over=False):
            best_bet_options.append((predictions["away_second_half_under"], confidence_score))

    # Most Likely Score Prediction with Probability Check
    most_likely_score_prob = score_probs.get(most_likely_score, 0)
    if most_likely_score_prob >= 0.5:  # At least 50% probability
        best_bet_options.append((f"Most Likely Score: {most_likely_score}", most_likely_score_prob))

    # Finalize Best Bet
    if not best_bet_options:
        best_bet = "Not enough data for reliable prediction"
        predictions["confidence_score"] = "0%"
    else:
        # Select the bet option with the highest score
        best_bet_options_sorted = sorted(best_bet_options, key=lambda x: x[1], reverse=True)
        best_bet = best_bet_options_sorted[0][0]  # Option with highest score

    # Populate final_predictions with all required fields
    final_predictions = {
        "predictions": predictions,
        "best_bet": best_bet,
        "best_bet_options": [option for option, score in best_bet_options],
        "monte_carlo_simulation": simulation_results,
        "avg_home_goals": average_home_goals,
        "avg_away_goals": average_away_goals,
        "home_win_streak": home_win_streak,
        "away_win_streak": away_win_streak,
        "total_goals_conceded_home": total_goals_conceded_home,
        "total_goals_conceded_away": total_goals_conceded_away,
        "confidence_score": f"{confidence_score:.2%}"
    }

    log_insights(event_data, predictions, confidence_score)

    return {
        "event": event_data,
        "final_predictions": final_predictions
    }
