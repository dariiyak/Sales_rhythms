import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def blend_predictions(
    p: np.ndarray,
    q: np.ndarray,
    prev_leftovers: np.ndarray | None = None,
    blend_alpha: float = 1.0,
    use_floor: bool = False,
) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    q = np.asarray(q, dtype="float64")

    y_pred = np.power(np.clip(p, 0.0, 1.0), blend_alpha) * q
    y_pred = np.clip(y_pred, 0.0, np.inf)

    if prev_leftovers is not None:
        cap = np.clip(np.asarray(prev_leftovers, dtype="float64"), 0.0, np.inf)
        y_pred = np.minimum(y_pred, cap)

    if use_floor:
        y_pred = np.floor(y_pred)

    return y_pred


def fit_stage1_classifier(
    train_part: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
) -> CatBoostClassifier:
    y_tr_cls = (train_part["qty"] > 0).astype("int8")
    w_tr_cls = np.where(y_tr_cls.values == 1, 7.0, 1.0)
    pool_tr_cls = Pool(train_part[features], label=y_tr_cls, weight=w_tr_cls, cat_features=cat_features)

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6,
        random_seed=42,
        verbose=200,
    )
    clf.fit(pool_tr_cls)
    return clf


def fit_stage2_regressor(
    train_part: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    quantile_alpha: float,
) -> CatBoostRegressor:
    tr_pos = train_part[train_part["qty"] > 0].copy()
    if tr_pos.empty:
        raise ValueError("Недостаточно положительных продаж для stage-2 регрессора.")

    w_tr_reg = np.sqrt(tr_pos["qty"].values)
    pool_tr_reg = Pool(tr_pos[features], label=tr_pos["qty"].values, weight=w_tr_reg, cat_features=cat_features)
    reg = CatBoostRegressor(
        loss_function=f"Quantile:alpha={quantile_alpha}",
        eval_metric=f"Quantile:alpha={quantile_alpha}",
        iterations=3200,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        random_seed=42,
        verbose=200,
    )
    reg.fit(pool_tr_reg)
    return reg


def predict_soft(
    df: pd.DataFrame,
    clf,
    reg,
    features: list[str],
    cat_features: list[str],
    blend_alpha: float = 1.0,
    use_floor: bool = False,
) -> np.ndarray:
    pool = Pool(df[features], cat_features=cat_features)

    p = clf.predict_proba(pool)[:, 1]
    q = reg.predict(pool)

    prev_leftovers = (
        df["prev_leftovers"].values.astype("float64")
        if "prev_leftovers" in df.columns
        else None
    )
    return blend_predictions(
        p=p,
        q=q,
        prev_leftovers=prev_leftovers,
        blend_alpha=blend_alpha,
        use_floor=use_floor,
    )
