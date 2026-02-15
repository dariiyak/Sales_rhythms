import numpy as np
import pandas as pd

from features import build_features, make_item_stats
from metrics import wmae
from modeling import fit_stage1_classifier, fit_stage2_regressor, predict_soft


def make_walk_forward_splits(
    train_df: pd.DataFrame,
    valid_days: int = 14,
    n_folds: int = 3,
    step_days: int = 14,
    min_train_days: int = 56,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
    max_dt = train_df["dt"].max()
    splits: list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]] = []

    for fold in range(n_folds):
        valid_end = max_dt - pd.Timedelta(days=fold * step_days)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
        train_end = valid_start - pd.Timedelta(days=1)

        tr = train_df[train_df["dt"] <= train_end].copy()
        va = train_df[(train_df["dt"] >= valid_start) & (train_df["dt"] <= valid_end)].copy()

        if tr.empty or va.empty:
            continue

        train_span = (tr["dt"].max() - tr["dt"].min()).days + 1
        if train_span < min_train_days:
            continue

        splits.append((tr, va, valid_start, valid_end))

    splits.sort(key=lambda x: x[2])
    return splits


def recursive_predict_period(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    item_stats: pd.DataFrame,
    clf,
    reg,
    features: list[str],
    cat_features: list[str],
    blend_alpha: float = 1.0,
    use_floor: bool = False,
) -> pd.DataFrame:
    history = history_df.copy()
    future = future_df.copy()

    history["__is_future"] = 0
    history["__future_row_id"] = -1

    future["__is_future"] = 1
    future["__future_row_id"] = np.arange(len(future), dtype=np.int64)
    future["qty"] = np.nan

    work = pd.concat([history, future], ignore_index=True)
    work = work.sort_values(["nm_id", "dt"]).reset_index(drop=True)
    work["__row_id"] = np.arange(len(work), dtype=np.int64)

    target_dates = np.sort(work.loc[work["__is_future"] == 1, "dt"].unique())
    for current_dt in target_dates:
        work_upto_dt = work[work["dt"] <= current_dt].copy()
        feat_work = build_features(work_upto_dt, item_stats=item_stats)
        day_mask = (feat_work["__is_future"] == 1) & (feat_work["dt"] == current_dt)

        if not day_mask.any():
            continue

        day_rows = feat_work.loc[day_mask]
        day_pred = predict_soft(
            day_rows,
            clf,
            reg,
            features,
            cat_features,
            blend_alpha=blend_alpha,
            use_floor=use_floor,
        )
        work.loc[day_rows["__row_id"].values, "qty"] = day_pred

    pred = work.loc[work["__is_future"] == 1, ["nm_id", "dt", "qty", "__future_row_id"]].copy()
    pred = pred.rename(columns={"qty": "pred_qty"})
    return pred


def score_recursive_fold(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    item_stats: pd.DataFrame,
    clf,
    reg,
    features: list[str],
    cat_features: list[str],
    blend_alpha: float,
    use_floor: bool,
) -> float:
    valid_pred_df = recursive_predict_period(
        history_df=train_raw,
        future_df=valid_raw,
        item_stats=item_stats,
        clf=clf,
        reg=reg,
        features=features,
        cat_features=cat_features,
        blend_alpha=blend_alpha,
        use_floor=use_floor,
    )
    eval_df = valid_raw[["nm_id", "dt", "qty"]].merge(
        valid_pred_df[["nm_id", "dt", "pred_qty"]],
        on=["nm_id", "dt"],
        how="left",
    )
    return wmae(eval_df["qty"].values, eval_df["pred_qty"].values)


def tune_two_stage_walk_forward_cv(
    train_df_raw: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    n_folds: int = 3,
    valid_days: int = 14,
    step_days: int = 14,
    min_train_days: int = 56,
    quantile_grid: list[float] | None = None,
    blend_alpha_grid: np.ndarray | None = None,
) -> tuple[float, float, bool, float]:
    if quantile_grid is None:
        quantile_grid = [0.6, 0.7, 0.8]
    if blend_alpha_grid is None:
        blend_alpha_grid = np.round(np.arange(0.7, 1.61, 0.1), 2)

    folds = make_walk_forward_splits(
        train_df=train_df_raw,
        valid_days=valid_days,
        n_folds=n_folds,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        raise ValueError("Не удалось сформировать walk-forward сплиты.")

    fold_artifacts = []
    for i, (tr_raw, va_raw, va_start, va_end) in enumerate(folds, start=1):
        item_stats = make_item_stats(tr_raw)
        va_hidden = va_raw.copy()
        va_hidden["qty"] = np.nan

        tune_all = pd.concat([tr_raw, va_hidden], ignore_index=True)
        tune_feat_all = build_features(tune_all, item_stats=item_stats)
        tr_feat = tune_feat_all[tune_feat_all["qty"].notna()].copy()

        clf = fit_stage1_classifier(tr_feat, features, cat_features)
        fold_artifacts.append(
            {
                "fold_id": i,
                "train_raw": tr_raw,
                "valid_raw": va_raw,
                "item_stats": item_stats,
                "train_feat": tr_feat,
                "clf": clf,
            }
        )
        print(
            f"[CV] fold={i} | valid={va_start.date()}..{va_end.date()} | "
            f"train_rows={len(tr_raw)} | valid_rows={len(va_raw)}"
        )

    quantile_scores: dict[float, float] = {}
    for quantile_alpha in quantile_grid:
        fold_scores = []
        for art in fold_artifacts:
            reg = fit_stage2_regressor(
                train_part=art["train_feat"],
                features=features,
                cat_features=cat_features,
                quantile_alpha=float(quantile_alpha),
            )
            score = score_recursive_fold(
                train_raw=art["train_raw"],
                valid_raw=art["valid_raw"],
                item_stats=art["item_stats"],
                clf=art["clf"],
                reg=reg,
                features=features,
                cat_features=cat_features,
                blend_alpha=1.0,
                use_floor=False,
            )
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        quantile_scores[float(quantile_alpha)] = mean_score
        print(f"[CV][Q] quantile={quantile_alpha:.2f} | mean_wMAE={mean_score:.6f}")

    best_quantile_alpha = min(quantile_scores, key=quantile_scores.get)

    for art in fold_artifacts:
        art["reg_best_q"] = fit_stage2_regressor(
            train_part=art["train_feat"],
            features=features,
            cat_features=cat_features,
            quantile_alpha=float(best_quantile_alpha),
        )

    best_blend_alpha = 1.0
    best_use_floor = False
    best_score = np.inf
    for blend_alpha in blend_alpha_grid:
        for use_floor in (False, True):
            fold_scores = []
            for art in fold_artifacts:
                score = score_recursive_fold(
                    train_raw=art["train_raw"],
                    valid_raw=art["valid_raw"],
                    item_stats=art["item_stats"],
                    clf=art["clf"],
                    reg=art["reg_best_q"],
                    features=features,
                    cat_features=cat_features,
                    blend_alpha=float(blend_alpha),
                    use_floor=use_floor,
                )
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            print(
                f"[CV][B] quantile={best_quantile_alpha:.2f} | "
                f"blend={blend_alpha:.2f} | floor={use_floor} | mean_wMAE={mean_score:.6f}"
            )

            if mean_score < best_score:
                best_score = mean_score
                best_blend_alpha = float(blend_alpha)
                best_use_floor = bool(use_floor)

    print(
        f"[CV][BEST] quantile={best_quantile_alpha:.2f} | "
        f"blend={best_blend_alpha:.2f} | floor={best_use_floor} | mean_wMAE={best_score:.6f}"
    )
    return best_quantile_alpha, best_blend_alpha, best_use_floor, best_score
