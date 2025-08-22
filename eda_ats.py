import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance

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


def fit_logistic_regression_nonlinear(dfx: pd.DataFrame):
    # Add nonlinearity via binning + one-hot of abs_spread and favorite-aligned edge
    if "week" not in dfx.columns:
        print("\n[LogReg+Bins] week not in data; skipping model.")
        return
    d = dfx.copy()
    d["favorite_is_home"] = d["favorite_is_home"].astype(int)
    for col in ["neutralSite", "conferenceGame"]:
        if col in d.columns:
            d[col] = d[col].astype("Int64").fillna(0).astype(int)
        else:
            d[col] = 0

    # Create buckets for nonlinearity
    spread_bins = [0, 3, 7, 14, 30]
    d["abs_spread_bucket"] = pd.cut(
        d["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    d["fav_edge_bin"] = pd.cut(
        d["fav_edge"], bins=edge_bins, right=False, include_lowest=True
    )

    # Combined interaction categorical: abs_spread_bucket × fav_edge_bin
    d["spread_edge_combo"] = (
        d["abs_spread_bucket"].astype(str) + "×" + d["fav_edge_bin"].astype(str)
    )

    # Model features
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
    X_num = d_model[numeric_features]
    X_cat = d_model[categorical_features]
    y = d_model["favorite_covered"].astype(int)

    # Chronological split
    train = d_model["week"] <= 10
    test = d_model["week"] >= 11
    if train.sum() == 0 or test.sum() == 0:
        print("\n[LogReg+Bins] Not enough data for chronological split; skipping.")
        return

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
            ("logreg", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    pipe.fit(d_model.loc[train, numeric_features + categorical_features], y[train])

    proba_train = pipe.predict_proba(
        d_model.loc[train, numeric_features + categorical_features]
    )[:, 1]
    proba_test = pipe.predict_proba(
        d_model.loc[test, numeric_features + categorical_features]
    )[:, 1]
    auc_train = roc_auc_score(y[train], proba_train)
    auc_test = roc_auc_score(y[test], proba_test)
    brier_train = brier_score_loss(y[train], proba_train)
    brier_test = brier_score_loss(y[test], proba_test)

    print("\n[LogReg+Bins] Numeric:", numeric_features)
    print("[LogReg+Bins] Categorical (one-hot):", categorical_features)
    print(f"[LogReg+Bins] Train AUC: {auc_train:.3f} | Test AUC: {auc_test:.3f}")
    print(
        f"[LogReg+Bins] Train Brier: {brier_train:.3f} | Test Brier: {brier_test:.3f}"
    )


def fit_logistic_regression_nonlinear_tuned(dfx: pd.DataFrame):
    # Grid over stronger regularization to reduce overfitting
    if "week" not in dfx.columns:
        return
    d = dfx.copy()
    d["favorite_is_home"] = d["favorite_is_home"].astype(int)
    for col in ["neutralSite", "conferenceGame"]:
        if col in d.columns:
            d[col] = d[col].astype("Int64").fillna(0).astype(int)
        else:
            d[col] = 0

    spread_bins = [0, 3, 7, 14, 30]
    d["abs_spread_bucket"] = pd.cut(
        d["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    d["fav_edge_bin"] = pd.cut(
        d["fav_edge"], bins=edge_bins, right=False, include_lowest=True
    )

    # Interaction combo used as categorical
    d["spread_edge_combo"] = (
        d["abs_spread_bucket"].astype(str) + "×" + d["fav_edge_bin"].astype(str)
    )

    numeric_features = [
        "elo_diff",
        "fav_edge",
        "abs_spread",
        "week",
        "favorite_is_home",
        "neutralSite",
        "conferenceGame",
    ]
    categorical_features = ["abs_spread_bucket", "fav_edge_bin", "spread_edge_combo"]
    d_model = d.dropna(
        subset=numeric_features + categorical_features + ["favorite_covered"]
    ).copy()
    y = d_model["favorite_covered"].astype(int)
    train = d_model["week"] <= 10
    test = d_model["week"] >= 11
    if train.sum() == 0 or test.sum() == 0:
        return

    Cs = [1.0, 0.5, 0.2, 0.1, 0.05]
    print("\n[LogReg+Bins Tuning] C values:", Cs)
    rows = []
    for C in Cs:
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
        pipe.fit(d_model.loc[train, numeric_features + categorical_features], y[train])
        proba_train = pipe.predict_proba(
            d_model.loc[train, numeric_features + categorical_features]
        )[:, 1]
        proba_test = pipe.predict_proba(
            d_model.loc[test, numeric_features + categorical_features]
        )[:, 1]
        auc_train = roc_auc_score(y[train], proba_train)
        auc_test = roc_auc_score(y[test], proba_test)
        brier_train = brier_score_loss(y[train], proba_train)
        brier_test = brier_score_loss(y[test], proba_test)
        rows.append(
            {
                "C": C,
                "train_auc": auc_train,
                "test_auc": auc_test,
                "train_brier": brier_train,
                "test_brier": brier_test,
            }
        )
    tune_df = pd.DataFrame(rows)
    print(
        tune_df.to_string(
            index=False,
            formatters={
                "train_auc": lambda v: f"{v:.3f}",
                "test_auc": lambda v: f"{v:.3f}",
                "train_brier": lambda v: f"{v:.3f}",
                "test_brier": lambda v: f"{v:.3f}",
            },
        )
    )


def fit_logreg_nonlinear_with_importance(dfx: pd.DataFrame, C: float = 0.5):
    # Fixed-C nonlinear model and permutation importances on test set
    if "week" not in dfx.columns:
        return
    d = dfx.copy()
    d["favorite_is_home"] = d["favorite_is_home"].astype(int)
    for col in ["neutralSite", "conferenceGame"]:
        if col in d.columns:
            d[col] = d[col].astype("Int64").fillna(0).astype(int)
        else:
            d[col] = 0

    spread_bins = [0, 3, 7, 14, 30]
    d["abs_spread_bucket"] = pd.cut(
        d["abs_spread"],
        bins=spread_bins,
        right=False,
        include_lowest=True,
        labels=["[0,3)", "[3,7)", "[7,14)", "[14,30)"],
    )
    edge_bins = [-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30]
    d["fav_edge_bin"] = pd.cut(
        d["fav_edge"], bins=edge_bins, right=False, include_lowest=True
    )
    d["spread_edge_combo"] = (
        d["abs_spread_bucket"].astype(str) + "×" + d["fav_edge_bin"].astype(str)
    )

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
    X = d_model[numeric_features + categorical_features]
    y = d_model["favorite_covered"].astype(int)
    train = d_model["week"] <= 10
    test = d_model["week"] >= 11
    if train.sum() == 0 or test.sum() == 0:
        return

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
    pipe.fit(X.loc[train], y[train])

    proba_train = pipe.predict_proba(X.loc[train])[:, 1]
    proba_test = pipe.predict_proba(X.loc[test])[:, 1]
    auc_train = roc_auc_score(y[train], proba_train)
    auc_test = roc_auc_score(y[test], proba_test)
    brier_train = brier_score_loss(y[train], proba_train)
    brier_test = brier_score_loss(y[test], proba_test)

    print(
        f"\n[LogReg+Bins C={C}] Train AUC: {auc_train:.3f} | Test AUC: {auc_test:.3f}"
    )
    print(
        f"[LogReg+Bins C={C}] Train Brier: {brier_train:.3f} | Test Brier: {brier_test:.3f}"
    )

    # Permutation importance on test set (AUC-based)
    result = permutation_importance(
        pipe, X.loc[test], y[test], scoring="roc_auc", n_repeats=20, random_state=42
    )
    importances = result.importances_mean
    features = list(X.columns)
    imp_df = pd.DataFrame(
        {"feature": features, "perm_auc_drop": importances}
    ).sort_values("perm_auc_drop", ascending=False)
    print(
        "\n[LogReg+Bins C={}] Permutation importance (AUC drop on test, top 15):".format(
            C
        )
    )
    print(
        imp_df.head(15).to_string(
            index=False, formatters={"perm_auc_drop": lambda v: f"{v:.4f}"}
        )
    )


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
    fit_logistic_regression_nonlinear(dfx)
    fit_logistic_regression_nonlinear_tuned(dfx)
    fit_logreg_nonlinear_with_importance(dfx, C=0.5)

    # Optional quick sanity print
    # print(f"Rows after filtering: {len(dfx)}")
    # print(f"Favorite cover rate: {dfx['favorite_covered'].mean():.3f}")

    # plot_abs_spread_hist(dfx, "plots/abs_spread_hist.png")
    # print("Saved: plots/abs_spread_hist.png")
