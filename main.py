import numpy as np
import pandas as pd

from cv_tuning import recursive_predict_period, tune_two_stage_walk_forward_cv
from features import build_features, make_item_stats
from metrics import wmae
from modeling import fit_stage1_classifier, fit_stage2_regressor


def main() -> None:
    train_path = "train.parquet"
    test_path = "test.parquet"
    sub_path = "sample_submission.csv"
    out_path = "submission.csv"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"] = pd.to_datetime(test["dt"])

    if "qty" not in train.columns:
        raise ValueError("В train нет колонки qty.")
    if "qty" in test.columns:
        test = test.drop(columns=["qty"])

    train = train.sort_values(["nm_id", "dt"]).reset_index(drop=True)
    test = test.sort_values(["nm_id", "dt"]).reset_index(drop=True)

    final_holdout_days = 14
    max_dt = train["dt"].max()
    holdout_start = max_dt - pd.Timedelta(days=final_holdout_days - 1)
    train_for_tuning = train[train["dt"] < holdout_start].copy()
    final_holdout = train[train["dt"] >= holdout_start].copy()

    if train_for_tuning.empty or final_holdout.empty:
        raise ValueError("Не удалось выделить финальный holdout из train.")

    item_stats_tune = make_item_stats(train_for_tuning)
    tune_train_feat = build_features(train_for_tuning, item_stats=item_stats_tune)
    drop_cols = {"qty", "dt", "__is_future", "__future_row_id", "__row_id"}
    features = [c for c in tune_train_feat.columns if c not in drop_cols]
    cat_features = ["nm_id"]

    # CV-тюнинг идет только на train_for_tuning (holdout не участвует)
    quantile_alpha, blend_alpha, use_floor, cv_score = tune_two_stage_walk_forward_cv(
        train_df_raw=train_for_tuning,
        features=features,
        cat_features=cat_features,
        n_folds=3,
        valid_days=14,
        step_days=14,
        min_train_days=56,
        quantile_grid=[0.6, 0.7, 0.8],
        blend_alpha_grid=np.round(np.arange(0.7, 1.61, 0.1), 2),
    )
    print(
        "CV tuned mean wMAE =",
        cv_score,
        "| Quantile alpha =",
        quantile_alpha,
        "| Blend alpha =",
        blend_alpha,
        "| floor =",
        use_floor,
    )

    # Честная проверка обобщения: один раз на отдельном holdout после тюнинга
    clf_tuned = fit_stage1_classifier(tune_train_feat, features, cat_features)
    reg_tuned = fit_stage2_regressor(
        tune_train_feat,
        features,
        cat_features,
        quantile_alpha=quantile_alpha,
    )
    holdout_pred_df = recursive_predict_period(
        history_df=train_for_tuning,
        future_df=final_holdout,
        item_stats=item_stats_tune,
        clf=clf_tuned,
        reg=reg_tuned,
        features=features,
        cat_features=cat_features,
        blend_alpha=blend_alpha,
        use_floor=use_floor,
    )
    holdout_eval = final_holdout[["nm_id", "dt", "qty"]].merge(
        holdout_pred_df[["nm_id", "dt", "pred_qty"]],
        on=["nm_id", "dt"],
        how="left",
    )
    holdout_score = wmae(holdout_eval["qty"].values, holdout_eval["pred_qty"].values)
    print(
        "Final holdout wMAE =",
        holdout_score,
        "| holdout from =",
        holdout_start.date(),
        "to",
        max_dt.date(),
    )

    # После оценки обучаем финальную модель на полном train
    item_stats_full = make_item_stats(train)
    full_train_feat = build_features(train, item_stats=item_stats_full)
    clf_full = fit_stage1_classifier(full_train_feat, features, cat_features)
    reg_full = fit_stage2_regressor(
        full_train_feat,
        features,
        cat_features,
        quantile_alpha=quantile_alpha,
    )

    test_pred_df = recursive_predict_period(
        history_df=train,
        future_df=test,
        item_stats=item_stats_full,
        clf=clf_full,
        reg=reg_full,
        features=features,
        cat_features=cat_features,
        blend_alpha=blend_alpha,
        use_floor=use_floor,
    )
    test_pred_df = test_pred_df.sort_values("__future_row_id").reset_index(drop=True)

    sub = pd.read_csv(sub_path)
    if {"nm_id", "dt"}.issubset(sub.columns):
        sub["dt"] = pd.to_datetime(sub["dt"])
        sub = sub.drop(columns=["qty"], errors="ignore")
        sub = sub.merge(
            test_pred_df[["nm_id", "dt", "pred_qty"]].rename(columns={"pred_qty": "qty"}),
            on=["nm_id", "dt"],
            how="left",
        )
    elif "id" in sub.columns and len(sub) == len(test_pred_df):
        sub["qty"] = test_pred_df["pred_qty"].values
    else:
        raise ValueError(
            "Не распознал формат sample_submission.csv. "
            "Покажи первые строки и названия колонок."
        )

    sub["qty"] = np.clip(sub["qty"].values.astype("float64"), 0.0, np.inf)
    sub.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
