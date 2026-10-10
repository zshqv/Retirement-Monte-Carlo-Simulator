import pandas as pd

RAW_PATH = "data/raw/chapt26.xlsx"
CLEAN_PATH = "data/shiller_real_returns.csv"


def load_shiller():
    raw = pd.read_excel(RAW_PATH, sheet_name=0, header=None, skiprows=8)
    df = raw.iloc[:, [0, 10, 14, 15]].copy()
    df.columns = ["year", "real_price", "real_dividend", "real_return"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def build_dataset():
    df = load_shiller()
    df = df.dropna(subset=["real_return"]).reset_index(drop=True)
    return df


def check_return_column(df_full):
    """Recompute the return from real price and real dividend, and compare."""
    implied = (df_full["real_price"].shift(-1) + df_full["real_dividend"]) / df_full["real_price"] - 1
    diff = (implied - df_full["real_return"]).abs()
    return diff.dropna().max()


def save_dataset(df, path=CLEAN_PATH):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    full = load_shiller()
    print("max difference, implied vs column:", check_return_column(full))

    df = build_dataset()
    print("rows:", len(df))
    print("years:", df["year"].min(), "to", df["year"].max())
    print("mean:", round(df["real_return"].mean(), 4))
    print("std:", round(df["real_return"].std(), 4))

    save_dataset(df)
    print("saved", len(df), "rows to", CLEAN_PATH)