import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["dt"].dt.dayofweek.astype("int16")
    df["weekofyear"] = df["dt"].dt.isocalendar().week.astype("int16")
    df["month"] = df["dt"].dt.month.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def add_group_lags_rollings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in [1, 2, 7, 14]:
        df[f"qty_lag_{lag}"] = df.groupby("nm_id")["qty"].shift(lag)

    g_qty = df.groupby("nm_id")["qty"]
    df["qty_roll_mean_7"] = g_qty.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["qty_roll_mean_14"] = g_qty.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
    df["qty_roll_median_7"] = g_qty.shift(1).rolling(7).median().reset_index(level=0, drop=True)
    df["qty_roll_max_14"] = g_qty.shift(1).rolling(14).max().reset_index(level=0, drop=True)

    qty_pos = (df["qty"] > 0).astype("float32")
    df["qty_pos_lag_1"] = qty_pos.groupby(df["nm_id"]).shift(1)
    df["qty_pos_roll_7"] = qty_pos.groupby(df["nm_id"]).shift(1).rolling(7).sum().reset_index(level=0, drop=True)
    df["qty_pos_roll_14"] = qty_pos.groupby(df["nm_id"]).shift(1).rolling(14).sum().reset_index(level=0, drop=True)

    last_sale_dt = (
        df["dt"]
        .where(df["qty"] > 0)
        .groupby(df["nm_id"])
        .shift(1)
        .groupby(df["nm_id"])
        .ffill()
    )
    df["days_since_last_sale"] = (df["dt"] - last_sale_dt).dt.days.astype("float32")

    for lag in [1, 7, 14]:
        df[f"price_lag_{lag}"] = df.groupby("nm_id")["price"].shift(lag)

    df["price_change_1"] = df["price"] / df["price_lag_1"] - 1.0
    df["price_change_7"] = df["price"] / df["price_lag_7"] - 1.0

    g_price = df.groupby("nm_id")["price"]
    df["price_roll_mean_28"] = g_price.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["price_rel"] = df["price"] / df["price_roll_mean_28"]

    for lag in [1, 7]:
        df[f"leftovers_lag_{lag}"] = df.groupby("nm_id")["prev_leftovers"].shift(lag)

    df["leftovers_change_1"] = df["prev_leftovers"] / df["leftovers_lag_1"] - 1.0
    df["stockout_flag"] = (df["prev_leftovers"] <= 0).astype("int8")
    df["low_stock_flag"] = (df["prev_leftovers"] < (df["qty_roll_mean_7"].fillna(0) * 0.7)).astype("int8")

    df["promo_x_price_rel"] = df["is_promo"].astype("float32") * df["price_rel"]
    df["promo_x_price_change_7"] = df["is_promo"].astype("float32") * df["price_change_7"]

    return df


def make_item_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    item_stats = (
        train_df.groupby("nm_id")
        .agg(
            item_mean_qty=("qty", "mean"),
            item_median_qty=("qty", "median"),
            item_pos_rate=("qty", lambda x: (x > 0).mean()),
            item_mean_price=("price", "mean"),
            item_promo_rate=("is_promo", "mean"),
        )
        .reset_index()
    )
    return item_stats


def add_item_aggregates(df_all: pd.DataFrame, item_stats: pd.DataFrame) -> pd.DataFrame:
    return df_all.merge(item_stats, on="nm_id", how="left")


def build_features(df_all: pd.DataFrame, item_stats: pd.DataFrame) -> pd.DataFrame:
    out = df_all.sort_values(["nm_id", "dt"]).reset_index(drop=True).copy()
    out = add_time_features(out)
    out = add_group_lags_rollings(out)
    out = add_item_aggregates(out, item_stats)
    return out
