"""
load_data.py
============
Handles all data loading, merging, and weekly aggregation for the
H&M Personalized Fashion Recommendations dataset.

FASHION INDUSTRY CONTEXT
-------------------------
H&M operates on a fast fashion model — new collections drop frequently,
SKU counts are massive, and transaction data reflects the combined signal
of trend adoption, seasonal demand, and promotional activity. Loading
this data correctly is the foundation of everything else.

The full dataset contains 31.7M transactions across 2 years (Sept 2018
to Sept 2020). We load all of it to capture full seasonal cycles — a
Prophet model needs at least 2 full cycles of yearly seasonality to
produce reliable forecasts. Loading only a sample produces flat, useless
forecasts because the model never sees a complete seasonal pattern.

WHY THESE FOUR CATEGORIES
--------------------------
Trousers   -> High volume (5.4M transactions), mild seasonality.
              Represents core wardrobe staples — hardest to forecast
              because demand is relatively stable year-round.

T-shirt    -> Strong summer peak, clear inverse of Sweater.
              Fast fashion's highest-velocity category globally.

Sweater    -> Sharpest seasonal pattern of all four. Demand spikes
              in Sept-Oct and collapses in spring. Best category for
              demonstrating Prophet's yearly seasonality component.

Swimwear   -> Most extreme seasonality — near zero in winter, massive
              spring/summer spike. Best candidate for Google Trends
              leading indicator because consumers search before they buy.
"""

import os
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


# ── Category Definitions ──────────────────────────────────────────────────────

# Maps human-readable category names to the exact column and value
# used in H&M's articles.csv. Confirmed via EDA.
#
# product_type_name is more granular than product_group_name.
# Swimwear uses product_group_name because swimwear types (bikini,
# swimsuit etc.) are split across multiple product_type_name values
# but all share the same product_group_name.

CATEGORIES = {
    'Trousers': ('product_type_name',  'Trousers'),
    'T-shirt':  ('product_type_name',  'T-shirt'),
    'Sweater':  ('product_type_name',  'Sweater'),
    'Swimwear': ('product_group_name', 'Swimwear'),
}


# ── Load Raw Data ─────────────────────────────────────────────────────────────

def load_transactions():
    """
    Loads the full H&M transactions dataset (31.7M rows).

    We load the full dataset — not a sample — because Prophet requires
    at least 2 full seasonal cycles to reliably decompose yearly
    seasonality. The dataset covers Sept 2018 to Sept 2020, giving us
    exactly 2 years. Loading fewer rows truncates the date range and
    produces flat, meaningless forecasts.

    Columns loaded:
        t_dat       : transaction date (parsed as datetime)
        customer_id : anonymized customer identifier
        article_id  : H&M article identifier (links to articles.csv)
        price       : normalized price

    Returns
    -------
    pd.DataFrame with 31.7M rows and 4 columns.
    Date range: 2018-09-20 to 2020-09-22.
    """
    path = os.path.join(RAW_DIR, 'transactions_train.csv')

    print("Loading transactions (31.7M rows)...")
    df = pd.read_csv(
        path,
        usecols=['t_dat', 'customer_id', 'article_id', 'price'],
        parse_dates=['t_dat']
    )
    print(f"Loaded: {df.shape[0]:,} rows")
    print(f"Date range: {df['t_dat'].min().date()} to {df['t_dat'].max().date()}")
    return df


def load_articles():
    """
    Loads the H&M articles metadata (105,542 unique articles).

    Key columns used downstream:
        article_id          : joins to transactions
        product_type_name   : granular category (Trousers, T-shirt, Sweater)
        product_group_name  : broad category (Swimwear, Garment Upper body)
        department_name     : H&M department
        index_group_name    : Ladies, Mens, Kids, Sport etc.
        garment_group_name  : Knitwear, Trousers, Jersey Basic etc.

    Returns
    -------
    pd.DataFrame with 105,542 rows and 25 columns.
    """
    path = os.path.join(RAW_DIR, 'articles.csv')

    print("Loading articles metadata...")
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} articles, {df.shape[1]} columns")
    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def load_merged():
    """
    Merges transactions with article metadata.

    Uses a left join on article_id — keeps all transactions and
    attaches category labels from articles.csv. Any transaction
    with an article_id not in articles.csv gets NaN category values
    which are filtered out when building category-specific series.

    Returns
    -------
    pd.DataFrame with 31.7M rows and 9 columns.
    """
    transactions = load_transactions()
    articles     = load_articles()

    print("Merging...")
    df = transactions.merge(
        articles[[
            'article_id',
            'product_type_name',
            'product_group_name',
            'department_name',
            'index_group_name',
            'garment_group_name'
        ]],
        on='article_id',
        how='left'
    )
    print(f"Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ── Build Weekly Time Series ──────────────────────────────────────────────────

def build_weekly_series(df, categories=CATEGORIES):
    """
    Aggregates merged transaction data into weekly sales volume
    for each target category.

    WHAT THIS PRODUCES
    ------------------
    For each category, a DataFrame with two columns:
        ds : week-ending date (Sunday), datetime
        y  : number of transactions in that week, integer

    This is the exact format Prophet requires. Prophet calls these
    columns ds (datestamp) and y (the value to forecast) by convention.

    WHY WEEKLY NOT DAILY
    --------------------
    Daily data in fashion is too noisy — individual days spike on
    weekends, promotions, and paydays. Weekly aggregation smooths
    this noise while preserving the seasonal patterns that matter
    for demand planning. This aligns with how H&M's actual planning
    teams operate — assortment and replenishment decisions are made
    on weekly cycles, not daily ones.

    WHY COUNT TRANSACTIONS NOT SUM REVENUE
    ---------------------------------------
    Transaction count measures raw demand volume — how many people
    bought something in this category this week. Revenue would conflate
    price changes with demand changes. A markdown event that doubles
    transaction volume while halving price shows flat revenue but the
    demand signal is a spike. We want the demand signal.

    Parameters
    ----------
    df         : merged DataFrame from load_merged()
    categories : dict mapping category name to (column, value) filter

    Returns
    -------
    dict of {category_name: pd.DataFrame} with ds and y columns.
    """
    weekly_series = {}

    for name, (col, val) in categories.items():
        series = (
            df[df[col] == val]
            .groupby(pd.Grouper(key='t_dat', freq='W'))
            .size()
            .reset_index(name='y')
            .rename(columns={'t_dat': 'ds'})
        )

        n_weeks    = len(series)
        n_missing  = series['y'].isna().sum()
        n_zero     = (series['y'] == 0).sum()
        avg_weekly = series['y'].mean()

        print(f"{name:12s}: {n_weeks} weeks | "
              f"avg {avg_weekly:,.0f} tx/week | "
              f"{n_missing} missing | {n_zero} zero weeks")

        weekly_series[name] = series

    return weekly_series


# ── Save Processed Data ───────────────────────────────────────────────────────

def save_weekly_series(weekly_series):
    """
    Saves each weekly series to data/processed/ as a CSV.
    These are the clean, model-ready files — not the raw source data.
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    for name, series in weekly_series.items():
        filename = f"weekly_{name.lower().replace('-', '_')}.csv"
        path     = os.path.join(PROC_DIR, filename)
        series.to_csv(path, index=False)
        print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Run this file directly to load data and build all weekly series.
    Output saved to data/processed/ for use by forecast.py and trends.py.

    Usage:
        python python/load_data.py
    """
    df            = load_merged()
    weekly_series = build_weekly_series(df)
    save_weekly_series(weekly_series)

    print("\nData pipeline complete.")
    print(f"Processed files saved to: {PROC_DIR}")