import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CSV_PATH = "data/cfbd_games_2024_with_closing.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_ats_target(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with a valid closing spread and ATS result (no pushes)
    mask_valid = (
        df["closing_spread"].notna()
        & df["favorite"].isin(["home", "away"])
        & df["ats_result"].isin(["home_cover", "away_cover"])
    )
    dfx = df.loc[mask_valid].copy()

    # favorite_covered = True if favorite side covered
    dfx["favorite_covered"] = (
        (dfx["favorite"] == "home") & (dfx["ats_result"] == "home_cover")
    ) | ((dfx["favorite"] == "away") & (dfx["ats_result"] == "away_cover"))

    # Simple feature for first plot
    dfx["abs_spread"] = dfx["closing_spread"].abs()
    return dfx


def plot_abs_spread_hist(dfx: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    # Bin edges chosen to be readable; adjust later as needed
    bins = [0, 1, 3, 5, 7, 10, 14, 21, 28, 35]
    plt.hist(dfx["abs_spread"], bins=bins, edgecolor="black")
    plt.title("Distribution of Absolute Closing Spread (2024)")
    plt.xlabel("|closing_spread| (points)")
    plt.ylabel("Game count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def summarize_cover_by_abs_spread(dfx: pd.DataFrame) -> pd.DataFrame:
    # Reuse the same bins you used for the histogram (adjust later if needed)
    bins = [0, 1, 3, 5, 7, 10, 14, 21, 28, 35]
    dfx = dfx.copy()
    dfx["abs_spread_bin"] = pd.cut(
        dfx["abs_spread"], bins=bins, right=False, include_lowest=True
    )

    g = dfx.groupby("abs_spread_bin", dropna=False)
    out = g.agg(
        n=("favorite_covered", "size"),
        covers=("favorite_covered", "sum"),
    ).reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    return out


def add_basic_features(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    dfx["favorite_is_home"] = dfx["favorite"] == "home"
    # Keep neutralSite / conferenceGame if present in your CSV
    for col in ["neutralSite", "conferenceGame", "line_provider"]:
        if col not in dfx.columns:
            dfx[col] = pd.NA
    return dfx


def summarize_cover_by(dfx: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    g = dfx.groupby(col, dropna=False)["favorite_covered"]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    # Optional: filter small groups
    return out[out["n"] >= min_n].sort_values("n", ascending=False)


def add_elo_features(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    # Keep rows with ELO available
    for col in ["homePregameElo", "awayPregameElo"]:
        if col not in dfx.columns:
            dfx[col] = pd.NA

    dfx = dfx[dfx["homePregameElo"].notna() & dfx["awayPregameElo"].notna()].copy()

    # ELO difference from home perspective (home - away)
    dfx["elo_diff"] = dfx["homePregameElo"] - dfx["awayPregameElo"]
    dfx["abs_elo_diff"] = dfx["elo_diff"].abs()

    # Market’s expected home margin is -closing_spread (home - away, implied by the spread)
    dfx["market_home_margin_exp"] = -dfx["closing_spread"]

    # Whether ELO and market agree on the favored side (sign alignment)
    dfx["same_side_elo_market"] = (
        (dfx["elo_diff"] > 0) & (dfx["market_home_margin_exp"] > 0)
    ) | ((dfx["elo_diff"] < 0) & (dfx["market_home_margin_exp"] < 0))

    return dfx


def summarize_cover_by_elo_bins(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    # ELO diffs can be wide; start with coarse bins, adjust later
    bins = [-400, -200, -100, -60, -40, -20, -10, 0, 10, 20, 40, 60, 100, 200, 400]
    dfx["elo_diff_bin"] = pd.cut(
        dfx["elo_diff"], bins=bins, right=False, include_lowest=True
    )
    g = dfx.groupby("elo_diff_bin")["favorite_covered"]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    return out


def summarize_cover_by_same_side(dfx: pd.DataFrame) -> pd.DataFrame:
    g = dfx.groupby("same_side_elo_market")["favorite_covered"]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    return out.sort_values("same_side_elo_market", ascending=False)


def add_elo_edge(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    dfx["elo_edge"] = dfx["elo_diff"] - dfx["market_home_margin_exp"]
    return dfx


def summarize_cover_by_elo_edge_bins(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    bins = [-30, -20, -10, -6, -3, -1, 0, 1, 3, 6, 10, 20, 30]
    dfx["elo_edge_bin"] = pd.cut(
        dfx["elo_edge"], bins=bins, right=False, include_lowest=True
    )
    g = dfx.groupby("elo_edge_bin", observed=True)["favorite_covered"]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    return out


def add_favorite_aligned_edge(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    # elo_edge is home-perspective; flip sign when favorite is away
    mult = dfx["favorite"].map({"home": 1, "away": -1})
    dfx["fav_edge"] = dfx["elo_edge"] * mult
    return dfx


def summarize_cover_by_fav_edge_bins(dfx: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    dfx = dfx.copy()
    bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    dfx["fav_edge_bin"] = pd.cut(
        dfx["fav_edge"], bins=bins, right=False, include_lowest=True
    )
    g = dfx.groupby("fav_edge_bin", observed=True)["favorite_covered"]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    return out[out["n"] >= min_n]


def summarize_cover_by_fav_edge_within_abs_spread(
    dfx: pd.DataFrame, min_n: int = 5
) -> pd.DataFrame:
    dfx = dfx.copy()
    # Define abs_spread buckets (match histogram bins roughly)
    spread_bins = [0, 3, 7, 14, 30]
    dfx["abs_spread_bucket"] = pd.cut(
        dfx["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )
    # Favorite-edge bins (reuse current bins)
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    dfx["fav_edge_bin"] = pd.cut(
        dfx["fav_edge"], bins=edge_bins, right=False, include_lowest=True
    )

    g = dfx.groupby(["abs_spread_bucket", "fav_edge_bin"], observed=True)[
        "favorite_covered"
    ]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    # Filter small cells
    out = out[out["n"] >= min_n]
    # Sort for readability
    return out.sort_values(["abs_spread_bucket", "fav_edge_bin"], ascending=True)


def summarize_cover_by_fav_edge_within_abs_spread_coarse(
    dfx: pd.DataFrame, min_n: int = 10
) -> pd.DataFrame:
    dfx = dfx.copy()
    spread_bins = [0, 3, 7, 14, 30]
    dfx["abs_spread_bucket"] = pd.cut(
        dfx["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )

    def edge_band(x: float) -> str:
        try:
            if x < -3:
                return "neg(<-3)"
            if x > 3:
                return "pos(>3)"
            return "near(±3)"
        except Exception:
            return "unknown"

    dfx["fav_edge_coarse"] = dfx["fav_edge"].apply(edge_band)
    g = dfx.groupby(["abs_spread_bucket", "fav_edge_coarse"], observed=True)[
        "favorite_covered"
    ]
    out = g.agg(n="size", covers="sum").reset_index()
    out["cover_rate"] = out["covers"] / out["n"]
    out = out[out["n"] >= min_n]
    return out.sort_values(["abs_spread_bucket", "fav_edge_coarse"], ascending=True)


def fit_logistic_regression(dfx: pd.DataFrame):
    # Chronological split by week
    if "week" not in dfx.columns:
        print("\n[LogReg] week not in data; skipping model.")
        return
    d = dfx.copy()
    # Feature matrix
    d["favorite_is_home"] = d["favorite_is_home"].astype(int)
    for col in ["neutralSite", "conferenceGame"]:
        if col in d.columns:
            d[col] = d[col].astype("Int64").fillna(0).astype(int)
        else:
            d[col] = 0
    # Select features
    feats = [
        "abs_spread",
        "favorite_is_home",
        "neutralSite",
        "conferenceGame",
        "week",
        "elo_diff",
        "fav_edge",
    ]
    d_model = d.dropna(subset=feats + ["favorite_covered"]).copy()
    X = d_model[feats]
    y = d_model["favorite_covered"].astype(int)

    train = d_model["week"] <= 10
    test = d_model["week"] >= 11
    if train.sum() == 0 or test.sum() == 0:
        print("\n[LogReg] Not enough data for chronological split; skipping.")
        return

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("logreg", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    pipe.fit(X[train], y[train])

    # Metrics
    proba_train = pipe.predict_proba(X[train])[:, 1]
    proba_test = pipe.predict_proba(X[test])[:, 1]
    auc_train = roc_auc_score(y[train], proba_train)
    auc_test = roc_auc_score(y[test], proba_test)
    brier_train = brier_score_loss(y[train], proba_train)
    brier_test = brier_score_loss(y[test], proba_test)

    print("\n[LogReg] Features:", feats)
    print(f"[LogReg] Train AUC: {auc_train:.3f} | Test AUC: {auc_test:.3f}")
    print(f"[LogReg] Train Brier: {brier_train:.3f} | Test Brier: {brier_test:.3f}")

    # Coefficients
    coef = pipe.named_steps["logreg"].coef_[0]
    intercept = pipe.named_steps["logreg"].intercept_[0]
    coef_table = pd.DataFrame({"feature": feats, "coef": coef}).sort_values("coef")
    print("\n[LogReg] Intercept:", round(intercept, 4))
    print("[LogReg] Coefficients:")
    print(coef_table.to_string(index=False))


if __name__ == "__main__":
    df = load_data(CSV_PATH)
    dfx = build_ats_target(df)
    dfx = add_basic_features(dfx)
    dfx = add_elo_features(dfx)
    dfx = add_elo_edge(dfx)
    dfx = add_favorite_aligned_edge(dfx)

    print("\nCover rate by favorite_is_home:")
    print(summarize_cover_by(dfx, "favorite_is_home").to_string(index=False))

    print("\nCover rate by neutralSite:")
    print(summarize_cover_by(dfx, "neutralSite").to_string(index=False))

    print("\nCover rate by conferenceGame:")
    print(summarize_cover_by(dfx, "conferenceGame").to_string(index=False))

    print("\nCover rate by line_provider (min 50):")
    print(summarize_cover_by(dfx, "line_provider", min_n=50).to_string(index=False))

    print("\nCover rate by ELO difference (home - away) bin:")
    print(summarize_cover_by_elo_bins(dfx).to_string(index=False))

    print("\nCover rate when ELO and market favor the same side:")
    print(summarize_cover_by_same_side(dfx).to_string(index=False))

    corr = (
        dfx[["elo_diff", "market_home_margin_exp"]].corr(method="spearman").iloc[0, 1]
    )
    print(f"\nSpearman correlation between ELO diff and market home margin: {corr:.3f}")

    summary = summarize_cover_by_abs_spread(dfx)
    print("\nATS cover rate by |closing_spread| bin:")
    print(summary.to_string(index=False))

    print("\nCover rate by ELO minus market edge (home perspective):")
    print(summarize_cover_by_elo_edge_bins(dfx).to_string(index=False))

    print("\nCover rate by favorite-aligned ELO edge:")
    print(summarize_cover_by_fav_edge_bins(dfx, min_n=5).to_string(index=False))

    print("\nCover rate by favorite-aligned ELO edge within |closing_spread| buckets:")
    print(
        summarize_cover_by_fav_edge_within_abs_spread(dfx, min_n=5).to_string(
            index=False
        )
    )

    print(
        "\n[Coarse] Cover rate by favorite-aligned ELO edge within |closing_spread| buckets:"
    )
    print(
        summarize_cover_by_fav_edge_within_abs_spread_coarse(dfx, min_n=10).to_string(
            index=False
        )
    )

    fit_logistic_regression(dfx)

    # Optional quick sanity print
    # print(f"Rows after filtering: {len(dfx)}")
    # print(f"Favorite cover rate: {dfx['favorite_covered'].mean():.3f}")

    # plot_abs_spread_hist(dfx, "plots/abs_spread_hist.png")
    # print("Saved: plots/abs_spread_hist.png")
