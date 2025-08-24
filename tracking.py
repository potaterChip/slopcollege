import os
import math
import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import cfbd

from predict_one import predict_cover_probability


LOG_PATH = "data/prediction_log.csv"
AMERICAN_MINUS_110_WIN = 100 / 110  # 0.909090...


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _favorite_from_spread(spread: float) -> str:
    if spread < 0:
        return "home"
    if spread > 0:
        return "away"
    return "none"


def _recommend_side(p_fav: float, threshold: float = 0.5238) -> str:
    # -110 breakeven ≈ 0.5238. Favor the dog if the model is <= (1 - threshold).
    if p_fav >= threshold:
        return "favorite"
    if p_fav <= 1 - threshold:
        return "underdog"
    return "pass"


def _profit_units(rec: str, favorite_covered: Optional[int]) -> float:
    if rec == "pass" or favorite_covered is None:
        return 0.0
    return (
        AMERICAN_MINUS_110_WIN
        if (
            (rec == "favorite" and favorite_covered == 1)
            or (rec == "underdog" and favorite_covered == 0)
        )
        else -1.0
    )


def _bet_result(rec: str, favorite_covered: Optional[int]) -> str:
    if rec == "pass":
        return "none"
    if favorite_covered is None:
        return ""
    if rec == "favorite":
        return "win" if favorite_covered == 1 else "loss"
    if rec == "underdog":
        return "win" if favorite_covered == 0 else "loss"
    return ""


def _brier(y: Optional[int], p: float) -> Optional[float]:
    return (y - p) ** 2 if y is not None else None


def _log_loss(y: Optional[int], p: float) -> Optional[float]:
    if y is None:
        return None
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _ensure_log_exists():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        cols = [
            "when_predicted_utc",
            "season",
            "week",
            "home_team",
            "away_team",
            "neutral_site",
            "conference_game",
            "spread_used",
            "favorite_side",
            "favorite_team",
            "pred_fav_cover_prob",
            "rec",
            "stake_units",
            # realized fields
            "home_points",
            "away_points",
            "ats_result",
            "favorite_covered",
            "correct",
            "brier",
            "log_loss",
            "bet_result",
            "profit_units",
        ]
        pd.DataFrame(columns=cols).to_csv(LOG_PATH, index=False)


def log_prediction(
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    spread: float,
    neutral_site: bool,
    conference_game: bool,
    C: float = 1.0,
    stake_units: float = 1.0,
    rec_threshold: float = 0.5238,
):
    _ensure_log_exists()

    p = predict_cover_probability(
        home_team=home_team,
        away_team=away_team,
        spread=spread,
        neutral_site=neutral_site,
        conference_game=conference_game,
        season=season,
        week=week,
        C=C,
    )
    favorite_side = _favorite_from_spread(spread)
    favorite_team = (
        home_team
        if favorite_side == "home"
        else away_team if favorite_side == "away" else ""
    )
    rec = _recommend_side(p, rec_threshold)
    stake = stake_units if rec != "pass" else 0.0

    row = {
        "when_predicted_utc": _now_iso(),
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "neutral_site": bool(neutral_site),
        "conference_game": bool(conference_game),
        "spread_used": float(spread),
        "favorite_side": favorite_side,
        "favorite_team": favorite_team,
        "pred_fav_cover_prob": float(p),
        "rec": rec,
        "stake_units": float(stake),
        # realized fields empty for now
        "home_points": None,
        "away_points": None,
        "ats_result": None,
        "favorite_covered": None,
        "correct": None,
        "brier": None,
        "log_loss": None,
        "bet_result": "none" if rec == "pass" else "",
        "profit_units": 0.0 if rec == "pass" else None,
    }
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(
        f"Logged prediction: {home_team} vs {away_team} (W{week} {season}) | p_fav_cover={p:.3f} | rec={rec}"
    )


def _get_cfbd_client():
    load_dotenv()
    token = (os.getenv("CFBD_API_KEY") or "").strip()
    if not token:
        raise RuntimeError("CFBD_API_KEY not set")
    cfg = cfbd.Configuration()
    cfg.access_token = token
    cfg.api_key["Authorization"] = token
    cfg.api_key_prefix["Authorization"] = "Bearer"
    return cfbd.ApiClient(cfg)


def update_results(season: int, week: int):
    _ensure_log_exists()
    df = pd.read_csv(LOG_PATH)

    # Select rows for this slate that are still missing outcomes
    mask = (
        (df["season"] == season)
        & (df["week"] == week)
        & (df["favorite_covered"].isna())
    )
    if not mask.any():
        print("No pending rows to update.")
        return

    with _get_cfbd_client() as client:
        games_api = cfbd.GamesApi(client)
        games = games_api.get_games(year=season, week=week)
        rows = []
        for g in games:
            d = g.to_dict()
            rows.append(
                {
                    "home_team": d.get("homeTeam"),
                    "away_team": d.get("awayTeam"),
                    "home_points": d.get("homePoints"),
                    "away_points": d.get("awayPoints"),
                }
            )
        gdf = pd.DataFrame(rows)

    # Merge by exact team strings
    m = df.merge(
        gdf, on=["home_team", "away_team"], how="left", suffixes=("", "_final")
    )

    # Compute realized fields for the target rows
    idx = m.index[
        m["season"].eq(season) & m["week"].eq(week) & m["favorite_covered"].isna()
    ]
    for i in idx:
        hp = m.at[i, "home_points_final"]
        ap = m.at[i, "away_points_final"]
        if pd.isna(hp) or pd.isna(ap):
            continue  # game not final yet

        spread = float(m.at[i, "spread_used"])  # home perspective
        fav_side = m.at[i, "favorite_side"]
        margin = float(hp) - float(ap)  # home margin

        # ATS grading relative to spread_used
        adj = margin + spread
        if adj > 0:
            ats_result = "home_cover"
        elif adj < 0:
            ats_result = "away_cover"
        else:
            ats_result = "push"

        y = (
            1
            if (
                (fav_side == "home" and ats_result == "home_cover")
                or (fav_side == "away" and ats_result == "away_cover")
            )
            else 0
        )
        if ats_result == "push":
            y = None

        p = float(
            m.at[i, "pred_fav_cover_prob"]
        )  # prediction was for favorite covering
        favorite_covered = y
        correct = (
            None
            if y is None or m.at[i, "rec"] == "pass"
            else int(
                (m.at[i, "rec"] == "favorite" and y == 1)
                or (m.at[i, "rec"] == "underdog" and y == 0)
            )
        )
        brier = _brier(y, p)
        log_loss = _log_loss(y, p)
        bet_result = _bet_result(m.at[i, "rec"], y)
        profit_units = _profit_units(m.at[i, "rec"], y)

        m.at[i, "home_points"] = hp
        m.at[i, "away_points"] = ap
        m.at[i, "ats_result"] = ats_result
        m.at[i, "favorite_covered"] = favorite_covered
        m.at[i, "correct"] = correct
        m.at[i, "brier"] = brier
        m.at[i, "log_loss"] = log_loss
        m.at[i, "bet_result"] = bet_result
        m.at[i, "profit_units"] = profit_units

    # Clean columns and save
    out = m[df.columns]  # keep original column order
    out.to_csv(LOG_PATH, index=False)
    print(f"Updated results for W{week} {season}.")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track ATS prediction logs and update with realized outcomes."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    plog = sub.add_parser("log", help="Log a single game prediction")
    plog.add_argument("--season", type=int, required=True)
    plog.add_argument("--week", type=int, required=True)
    plog.add_argument("--home", dest="home_team", type=str, required=True)
    plog.add_argument("--away", dest="away_team", type=str, required=True)
    plog.add_argument(
        "--spread",
        type=float,
        required=True,
        help="Home perspective spread (home negative = home favored)",
    )
    plog.add_argument("--neutral-site", action="store_true", help="Set if neutral site")
    plog.add_argument(
        "--conference-game", action="store_true", help="Set if conference game"
    )
    plog.add_argument("--C", type=float, default=1.0, help="Model regularization C")
    plog.add_argument("--stake", dest="stake_units", type=float, default=1.0)
    plog.add_argument(
        "--threshold",
        dest="rec_threshold",
        type=float,
        default=0.5238,
        help="Recommendation threshold (default ~-110 breakeven)",
    )

    pupd = sub.add_parser("update", help="Update realized results for a slate")
    pupd.add_argument("--season", type=int, required=True)
    pupd.add_argument("--week", type=int, required=True)

    return p


def main():
    parser = _build_cli()
    args = parser.parse_args()

    if args.cmd == "log":
        log_prediction(
            season=args.season,
            week=args.week,
            home_team=args.home_team,
            away_team=args.away_team,
            spread=args.spread,
            neutral_site=bool(args.neutral_site),
            conference_game=bool(args.conference_game),
            C=float(args.C),
            stake_units=float(args.stake_units),
            rec_threshold=float(args.rec_threshold),
        )
    elif args.cmd == "update":
        update_results(season=args.season, week=args.week)


if __name__ == "__main__":
    main()
