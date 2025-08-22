# train_model_nonlinear

```
mask_valid = (
        df["closing_spread"].notna()
        & df["favorite"].isin(["home", "away"])
        & df["ats_result"].isin(["home_cover", "away_cover"])
    )
d = df.loc[mask_valid].copy()
```

mask_valid result is a pandas series of booleans that have the same index and length as df. Either true or false based on the 3 criteria.
the index will not necessarily be in order anymore, but this shouldn't really matter with the model training. It only matters if you want neat 0..n row numbers for displays or if you
actually do something dependent on positional indices

```
d["market_home_margin_exp"] = -d["closing_spread"]
d["elo_edge"] = d["elo_diff"] - d["market_home_margin_exp"]
```

What market_home_margin_exp is
Definition: The market’s implied expected scoring margin for the home team, in points.
Computation: market_home_margin_exp = -closing_spread
Your closing_spread is from the home team’s perspective (negative = home favored).
Negating converts that to “how many points the home team is expected to win/lose by.”
Quick examples
closing_spread = -7 → market_home_margin_exp = +7 (home expected to win by 7)
closing_spread = +3.5 → market_home_margin_exp = -3.5 (home expected to lose by 3.5)
closing_spread = 0 → 0 (pick’em)
Why we use it
It lets us compare model strength to the market:
elo_edge = elo_diff - market_home_margin_exp
If positive, Elo thinks the home team should do better than the market implies.

## The model itself

```
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
```

Why StandardScaler
Normalize numeric scales: elo_diff, abs_spread, week, and binary flags are on different ranges. Scaling stabilizes optimization and makes regularization fair across features.
Works with sparse stacking: with_mean=False avoids centering (which would break sparsity) so it can stack with the sparse one-hot block.
Why OneHotEncoder
Turn categories into usable features: Linear models need numeric inputs; one-hot makes abs_spread_bucket, fav_edge_bin, spread_edge_combo into indicator columns.
No accidental ordinality: Buckets are not ordered for the model; one-hot avoids implying rank like an integer encoder would.
Safe for unseen values: handle_unknown="ignore" prevents errors at prediction time.
Memory-efficient: sparse_output=True keeps the huge, mostly-zero matrix efficient.
Why LogisticRegression
Right objective: We’re predicting a binary event (favorite covers), and we want probabilities.
Interpretable + regularized: Coefficients are understandable; C controls L2 regularization to prevent overfit.
Plays well with sparse OHE: Optimizers handle high-dimensional sparse inputs well; fast to train.
Strong baseline: Few knobs, robust, and easy to debug before trying complex models.
Why not something else (for now)
Tree/boosted models: Capture nonlinearity automatically but need more tuning and usually probability calibration; can overfit small signals.
SVM/NN: Heavier to tune, less interpretable, and not necessarily better for well-engineered tabular features.
If/when to revisit:
Add boosted trees (e.g., XGBoost/LightGBM) if you need more nonlinearity.
Add probability calibration (Platt/isotonic) if probability sharpness matters.
