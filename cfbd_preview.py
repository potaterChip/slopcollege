import os
import pandas as pd
from dotenv import load_dotenv
import cfbd


def get_cfbd_client():
    load_dotenv()
    token = (os.getenv("CFBD_API_KEY") or "").strip()
    if not token:
        raise RuntimeError("CFBD_API_KEY not set. Export it or put it in .env")
    # Configure both access_token and explicit header mapping for maximum compatibility
    cfg = cfbd.Configuration()
    cfg.access_token = token
    cfg.api_key["Authorization"] = token
    cfg.api_key_prefix["Authorization"] = "Bearer"
    return cfbd.ApiClient(cfg)


def fetch_games_2024(api_client) -> pd.DataFrame:
    games_api = cfbd.GamesApi(api_client)
    games = games_api.get_games(year=2024)
    return pd.DataFrame([g.to_dict() for g in games])


def fetch_lines_2024(api_client) -> pd.DataFrame:
    betting_api = cfbd.BettingApi(api_client)
    lines = betting_api.get_lines(year=2024)
    return pd.DataFrame([l.to_dict() for l in lines])


def flatten_closing_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        return pd.DataFrame(
            columns=["id", "line_provider", "closing_spread", "closing_total"]
        )

    # Ensure list-like
    lines_df = lines_df.copy()
    lines_df["lines"] = lines_df["lines"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    exploded = lines_df[["id", "lines"]].explode("lines", ignore_index=True)
    exploded = exploded.dropna(subset=["lines"])  # keep only rows with a line dict

    # Normalize line dicts into columns
    normalized = pd.json_normalize(exploded["lines"]).add_prefix("line.")
    # Combine with ids
    extracted_df = pd.concat(
        [exploded[["id"]].reset_index(drop=True), normalized.reset_index(drop=True)],
        axis=1,
    )

    # Provider name can be a plain string (CFBD) or nested
    def get_provider_name(row):
        prov = row.get("line.provider")
        if isinstance(prov, str):
            return prov
        if isinstance(prov, dict):
            return prov.get("name") or prov.get("displayName") or prov.get("id")
        return None

    extracted_df["line_provider"] = extracted_df.apply(get_provider_name, axis=1)

    # Extract spread and total from possible fields
    def coerce_num(val):
        return pd.to_numeric(val, errors="coerce")

    spread_cols = [
        c
        for c in extracted_df.columns
        if c.split(".")[-1]
        in ("spread", "formattedSpread", "closingSpread", "spread_close")
    ]
    total_cols = [
        c
        for c in extracted_df.columns
        if c.split(".")[-1] in ("overUnder", "total", "closingTotal", "total_close")
    ]
    if not spread_cols:
        spread_cols = ["line.spread"] if "line.spread" in extracted_df.columns else []
    if not total_cols:
        total_cols = (
            ["line.overUnder"] if "line.overUnder" in extracted_df.columns else []
        )

    def first_notna_numeric(row, cols):
        for c in cols:
            if c in row and pd.notna(row[c]):
                v = coerce_num(row[c])
                if pd.notna(v):
                    return v
        return pd.NA

    extracted_df["closing_spread"] = extracted_df.apply(
        lambda r: first_notna_numeric(r, spread_cols), axis=1
    )
    extracted_df["closing_total"] = extracted_df.apply(
        lambda r: first_notna_numeric(r, total_cols), axis=1
    )

    # Provider preference
    provider_rank = {
        None: 99,
        "consensus": 0,
        "Consensus": 0,
        "Vegas": 1,
        "Caesars": 2,
        "DraftKings": 3,
        "FanDuel": 4,
        "BetMGM": 5,
        "Pinnacle": 6,
    }
    extracted_df["_rank"] = extracted_df["line_provider"].map(provider_rank).fillna(50)

    # Choose best per game id; prefer having both values; then provider rank; then notna count
    extracted_df["_both"] = (
        extracted_df[["closing_spread", "closing_total"]].notna().sum(axis=1)
    )
    best = (
        extracted_df.sort_values(by=["_both", "_rank"], ascending=[False, True])
        .groupby("id", as_index=False)
        .first()[["id", "line_provider", "closing_spread", "closing_total"]]
    )
    return best


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    with get_cfbd_client() as client:
        games_df = fetch_games_2024(client)
        lines_df = fetch_lines_2024(client)

    games_path = "data/cfbd_games_2024.csv"
    lines_path = "data/cfbd_lines_2024.csv"
    games_df.to_csv(games_path, index=False)
    lines_df.to_csv(lines_path, index=False)

    # Flatten closing lines and merge
    closing_df = flatten_closing_lines(lines_df)
    merged = games_df.merge(closing_df, on="id", how="left")
    merged_path = "data/cfbd_games_2024_with_closing.csv"
    merged.to_csv(merged_path, index=False)

    print(f"Saved {len(games_df)} games -> {games_path}")
    print(f"Saved {len(lines_df)} line records -> {lines_path}")
    print(f"Saved merged with closing -> {merged_path}")
    print()
    print(
        merged[
            [
                "id",
                "homeTeam",
                "awayTeam",
                "closing_spread",
                "closing_total",
                "line_provider",
            ]
        ].head(10)
    )
