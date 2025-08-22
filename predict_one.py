import os
import pandas as pd
from dotenv import load_dotenv
import cfbd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def get_cfbd_client():
    load_dotenv()
    token = (os.getenv("CFBD_API_KEY") or "").strip()
    if not token:
        raise RuntimeError("CFBD_API_KEY not set. Export it or put it in .env")
    cfg = cfbd.Configuration()
    cfg.access_token = token
    cfg.api_key["Authorization"] = token
    cfg.api_key_prefix["Authorization"] = "Bearer"
    return cfbd.ApiClient(cfg)


def train_model_nonlinear(csv_path: str, C: float = 1.0) -> Pipeline:
    df = pd.read_csv(csv_path)
    # Build ATS target and features consistent with eda script
    mask_valid = (
        df["closing_spread"].notna()
        & df["favorite"].isin(["home", "away"])
        & df["ats_result"].isin(["home_cover", "away_cover"])
    )
    d = df.loc[mask_valid].copy()

    d["favorite_covered"] = (
        (d["favorite"] == "home") & (d["ats_result"] == "home_cover")
    ) | ((d["favorite"] == "away") & (d["ats_result"] == "away_cover"))

    d["abs_spread"] = d["closing_spread"].abs()

    d["favorite_is_home"] = (d["favorite"] == "home").astype(int)

    # ELO/edge features
    d = d[
        d["homePregameElo"].notna() & d["awayPregameElo"].notna()
    ].copy()  # another series filter like above

    d["elo_diff"] = d["homePregameElo"] - d["awayPregameElo"]

    d["market_home_margin_exp"] = -d["closing_spread"]

    d["elo_edge"] = d["elo_diff"] - d["market_home_margin_exp"]

    mult = d["favorite"].map({"home": 1, "away": -1})
    d["fav_edge"] = d["elo_edge"] * mult

    # Buckets and interaction
    spread_bins = [0, 3, 7, 14, 30]
    d["abs_spread_bucket"] = pd.cut(
        d["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    edge_labels = [
        "[-30,-12)",
        "[-12,-6)",
        "[-6,-3)",
        "[-3,-1)",
        "[-1,0)",
        "[0,1)",
        "[1,3)",
        "[3,6)",
        "[6,12)",
        "[12,30)",
    ]
    d["fav_edge_bin"] = pd.cut(
        d["fav_edge"],
        bins=edge_bins,
        right=False,
        include_lowest=True,
        labels=edge_labels,
    )
    d["neutralSite"] = d["neutralSite"].astype("Int64").fillna(0).astype(int)
    d["conferenceGame"] = d["conferenceGame"].astype("Int64").fillna(0).astype(int)
    d["spread_edge_combo"] = (
        d["abs_spread_bucket"].astype(str) + "×" + d["fav_edge_bin"].astype(str)
    )

    # Ensure categorical columns contain no NaNs (OneHotEncoder categories must be clean)
    for col in ["abs_spread_bucket", "fav_edge_bin", "spread_edge_combo"]:
        d[col] = d[col].astype("object").fillna("missing")

    numeric_features = [
        "elo_diff",
        "fav_edge",
        "abs_spread",
        "week",
        "favorite_is_home",
        "neutralSite",
        "conferenceGame",
    ]
    categorical_features = [
        "abs_spread_bucket",
        "fav_edge_bin",
        "spread_edge_combo",
    ]
    d_model = d.dropna(
        subset=numeric_features + categorical_features + ["favorite_covered"]
    ).copy()
    y = d_model["favorite_covered"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_features,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("logreg", LogisticRegression(max_iter=1000, solver="liblinear", C=C)),
        ]
    )
    pipe.fit(d_model[numeric_features + categorical_features], y)
    return pipe


def fetch_team_elo(
    elo_api: cfbd.RatingsApi, team: str, season: int, week: int
) -> float:
    # CFBD ratings Elo endpoint gives current ratings per week
    try:
        ratings = elo_api.get_elo(year=2024)
    except AttributeError:
        print(f"Error: API method not available in this client version")
        # Fallback: API method not available in this client version
        return float("nan")
    for r in ratings:
        if str(r.team).lower() == team.lower():
            return float(r.elo)
    return float("nan")


def predict_cover_probability(
    home_team: str,
    away_team: str,
    spread: float,
    neutral_site: bool,
    conference_game: bool,
    season: int,
    week: int,
    C: float = 1.0,
) -> float:
    # Train model on 2024 data
    model = train_model_nonlinear("data/cfbd_games_2024_with_closing.csv", C=C)

    # Build one-row input using Elo for given season/week
    with get_cfbd_client() as client:
        elo_api = cfbd.RatingsApi(client)
        home_elo = fetch_team_elo(elo_api, home_team, season, week)
        away_elo = fetch_team_elo(elo_api, away_team, season, week)
        print(f"Home Elo: {home_elo}, Away Elo: {away_elo}")

    # Features
    favorite = "home" if spread < 0 else "away" if spread > 0 else "none"
    favorite_is_home = 1 if favorite == "home" else 0
    abs_spread = abs(spread)
    if pd.notna(home_elo) and pd.notna(away_elo):
        elo_diff = home_elo - away_elo
    else:
        # Fallback if Elo not available: assume no Elo edge
        elo_diff = 0.0
    market_home_margin_exp = -spread
    elo_edge = elo_diff - market_home_margin_exp
    mult = 1 if favorite == "home" else -1 if favorite == "away" else 0
    fav_edge = elo_edge * mult

    spread_bins = [0, 3, 7, 14, 30]
    abs_spread_bucket_val = pd.cut(
        [abs_spread],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )[0]
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    edge_labels = [
        "[-30,-12)",
        "[-12,-6)",
        "[-6,-3)",
        "[-3,-1)",
        "[-1,0)",
        "[0,1)",
        "[1,3)",
        "[3,6)",
        "[6,12)",
        "[12,30)",
    ]
    fav_edge_bin_val = pd.cut(
        [fav_edge], bins=edge_bins, right=False, include_lowest=True, labels=edge_labels
    )[0]
    # Coerce to strings and handle missing
    abs_spread_bucket = (
        str(abs_spread_bucket_val) if pd.notna(abs_spread_bucket_val) else "missing"
    )
    fav_edge_bin = str(fav_edge_bin_val) if pd.notna(fav_edge_bin_val) else "missing"
    spread_edge_combo = f"{abs_spread_bucket}×{fav_edge_bin}"

    row = pd.DataFrame(
        {
            "elo_diff": [elo_diff],
            "fav_edge": [fav_edge],
            "abs_spread": [abs_spread],
            "week": [week],
            "favorite_is_home": [favorite_is_home],
            "neutralSite": [1 if neutral_site else 0],
            "conferenceGame": [1 if conference_game else 0],
            "abs_spread_bucket": [abs_spread_bucket],
            "fav_edge_bin": [fav_edge_bin],
            "spread_edge_combo": [spread_edge_combo],
        }
    )

    prob = float(model.predict_proba(row)[0, 1])
    return prob


if __name__ == "__main__":
    # Example: replace with CLI args if needed
    home_team = os.getenv("HOME_TEAM", "Kansas State")
    away_team = os.getenv("AWAY_TEAM", "Iowa State")
    spread = float(os.getenv("SPREAD", "-3"))
    neutral_site = os.getenv("NEUTRAL_SITE", "true").lower() == "true"
    conference_game = os.getenv("CONFERENCE_GAME", "true").lower() == "true"
    season = int(os.getenv("SEASON", "2025"))
    week = int(os.getenv("WEEK", "0"))
    C = float(os.getenv("C", "1.0"))

    p_cover = predict_cover_probability(
        home_team, away_team, spread, neutral_site, conference_game, season, week, C=C
    )
    favorite = "home" if spread < 0 else "away" if spread > 0 else "none"
    print(f"Favorite side: {favorite} | Predicted cover probability: {p_cover:.3f}")
