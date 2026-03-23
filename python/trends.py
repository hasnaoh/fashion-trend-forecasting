"""
trends.py
=========
Pulls Google Trends search volume data via pytrends and runs
cross-correlation analysis against H&M transaction data to test
whether search interest is a leading indicator of fashion demand.

THE RESEARCH QUESTION
----------------------
Does consumer search behavior on Google predict purchase behavior
at H&M 2-4 weeks in advance?

This is an operationally meaningful question. If search volume for
"swimwear" reliably peaks 3 weeks before H&M swimwear transactions
peak, then a demand planner could use Google Trends as a real-time
signal to adjust inventory positioning ahead of demand — reducing
stockouts during peak and reducing overstock during off-peak.

This is the "exogenous variable" layer from your research:
    "Social media and search trends: AI driven platforms monitor
    and search for rising keywords. Analyzing image content from
    influencers allows brands to detect micro-drops and adjust
    production before the trend reaches peak popularity."

Google Trends is the most accessible proxy for this signal and
has been used in academic fashion forecasting research as a
validated leading indicator for seasonal categories.

WHY THE US MARKET
-----------------
Google Trends is filtered to the United States (geo='US') while
the H&M transaction data is global. This creates a geographic
mismatch that is acknowledged as a limitation. The rationale:

1. The US is H&M's largest single market by revenue
2. US search trends reflect English-language fashion media which
   has global influence on trend adoption
3. US seasonal patterns (Northern Hemisphere) align with the
   dominant seasonal signal in H&M's global transaction data
   since the majority of H&M stores are in Europe and North America

TIMEFRAME ALIGNMENT
-------------------
Both datasets cover September 2018 to September 2020. The Google
Trends timeframe is set to match exactly so that cross-correlation
operates on synchronized time series.

CROSS-CORRELATION METHODOLOGY
------------------------------
Standard Pearson correlation measures whether two series move
together at the same time. Cross-correlation shifts one series
forward by k periods (lags) and measures correlation at each lag:

    correlation(k) = corr(sales(t), search(t - k))

This asks: does search volume from k weeks ago predict today's sales?

We test lags 0 through 8 weeks. The lag with the highest correlation
is the "leading indicator window" — the number of weeks in advance
that search interest predicts purchase volume.

INTERPRETING RESULTS
--------------------
Correlation thresholds (Pearson r):
    r >= 0.7  : Strong leading indicator — actionable for planning
    r 0.5-0.7 : Moderate — useful as one signal among several
    r 0.3-0.5 : Weak — limited practical value
    r < 0.3   : No meaningful relationship

P-value < 0.05 indicates the correlation is statistically significant
(less than 5% probability the relationship is random). Always reported
alongside r to prevent spurious correlation claims.

EXPECTED FINDINGS
-----------------
Based on the seasonal patterns observed in EDA:

Swimwear   -> Strongest leading indicator expected. Consumers research
              swimwear purchases weeks before buying. Search spikes in
              April-May, transactions peak in May-June.

Sweater    -> Strong seasonal signal but shorter lead. Search for
              winter clothing tends to spike close to purchase timing.
              Expected lag: 1-2 weeks.

T-shirt    -> Weaker relationship expected. T-shirt purchases are
              more impulse-driven and less researched online.

Trousers   -> Weakest relationship expected. Core wardrobe staple
              with low seasonal volatility — less consumer research
              behavior before purchase.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from pytrends.request import TrendReq

warnings.filterwarnings('ignore')

import os
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')


# ── Keyword Mapping ───────────────────────────────────────────────────────────

# Maps H&M category names to Google search terms.
#
# Keyword selection rationale:
# - Terms are broad enough to capture general category search behavior
# - Avoid brand-specific terms (not measuring H&M brand searches)
# - Terms reflect how US consumers actually search (pants not trousers,
#   since H&M uses British English in their data but US consumers
#   search in American English)
# - Max 5 keywords per pytrends request (API limit)

KEYWORDS = {
    'Trousers': ['pants', 'trousers', 'dress pants', 'wide leg pants', 'slim pants'],
    'T-shirt':  ['t-shirt', 'graphic tee', 'tshirt', 'oversized tshirt', 'basic tee'],
    'Sweater':  ['sweater', 'knit sweater', 'cardigan', 'chunky knit', 'winter sweater'],
    'Swimwear': ['swimsuit', 'bikini', 'swimwear', 'bathing suit', 'one piece swimsuit'],
}

# H&M data date range — Trends must match
TIMEFRAME = '2018-09-01 2020-09-22'
GEO       = 'US'


# ── Pull Google Trends ────────────────────────────────────────────────────────

def get_trends(keywords, timeframe=TIMEFRAME, geo=GEO, retries=3, delay=5):
    """
    Pulls weekly Google Trends interest for a list of keywords.

    Returns relative interest (0-100 scale) where 100 = peak search
    interest for that term over the selected period. This is NOT
    absolute search volume — it is normalized relative interest.

    This normalization is actually useful for cross-correlation:
    we are testing the shape and timing of the search pattern relative
    to the transaction pattern, not the absolute magnitude.

    Parameters
    ----------
    keywords  : list of up to 5 search terms
    timeframe : 'YYYY-MM-DD YYYY-MM-DD' string
    geo       : country code ('US', 'GB', '' for worldwide)
    retries   : number of retry attempts on rate limit
    delay     : seconds to wait between requests

    Returns
    -------
    pd.DataFrame with datetime index and one column per keyword.
    """
    pytrends = TrendReq(hl='en-US', tz=300)  # tz=300 for US Eastern

    for attempt in range(retries):
        try:
            pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo=geo
            )
            df = pytrends.interest_over_time()

            if df.empty:
                print(f"  No data returned for: {keywords}")
                return pd.DataFrame()

            df = df.drop(columns=['isPartial'], errors='ignore')
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            if attempt < retries - 1:
                print(f"  Rate limited, waiting {delay}s... (attempt {attempt+1})")
                time.sleep(delay)
            else:
                print(f"  Failed after {retries} attempts: {e}")
                return pd.DataFrame()


def get_all_trends(keywords_map=KEYWORDS, delay=3):
    """
    Pulls trends for all four categories with rate limit protection.

    pytrends can be rate limited if called too rapidly. A delay
    between requests prevents 429 errors.

    Returns
    -------
    dict of {category_name: pd.DataFrame with keyword columns}
    """
    all_trends = {}

    for category, keywords in keywords_map.items():
        print(f"Fetching trends for {category}...")
        df = get_trends(keywords)

        if not df.empty:
            # Create a single composite signal by averaging all keywords
            # This is more robust than using any single keyword alone
            df['composite'] = df.mean(axis=1)
            all_trends[category] = df
            print(f"  Retrieved {len(df)} weeks of data")
        else:
            print(f"  Skipping {category} — no data returned")

        time.sleep(delay)

    return all_trends


# ── Cross-Correlation Analysis ────────────────────────────────────────────────

def cross_correlate(sales_series, trends_df, max_lag=8):
    """
    Tests whether Google search volume leads H&M transaction volume.

    For each lag k (0 to max_lag weeks), shifts the trends series
    forward by k weeks and computes Pearson correlation with sales:

        r(k) = Pearson_r(sales(t), search(t - k))

    A high r at lag k=3 means: search volume from 3 weeks ago is
    a strong predictor of this week's transaction volume. This is
    the actionable finding — a demand planner checking Google Trends
    today has a 3-week window to respond before the demand spike hits.

    Parameters
    ----------
    sales_series : pd.DataFrame with ds and y columns
    trends_df    : pd.DataFrame with datetime index and composite column
    max_lag      : int, maximum weeks to test (default 8)

    Returns
    -------
    pd.DataFrame with columns: lag_weeks, correlation, pvalue, significant
    """
    # Align both series on a common weekly date index
    sales = sales_series.set_index('ds')['y']
    search = trends_df['composite'].resample('W').mean()

    # Find overlapping dates
    common_dates = sales.index.intersection(search.index)
    sales  = sales.loc[common_dates]
    search = search.loc[common_dates]

    results = []

    for lag in range(0, max_lag + 1):
        # Shift search forward by lag weeks
        # search(t - lag) predicts sales(t)
        shifted_search = search.shift(lag)

        # Drop NaN introduced by shift
        valid = ~shifted_search.isna()
        s     = sales[valid]
        t     = shifted_search[valid]

        if len(s) < 10:
            continue

        r, p = pearsonr(s, t)

        results.append({
            'lag_weeks':   lag,
            'correlation': round(r, 4),
            'pvalue':      round(p, 4),
            'significant': p < 0.05
        })

    return pd.DataFrame(results)


def find_best_lag(lag_results):
    """
    Returns the lag with the highest statistically significant correlation.
    If no significant lags, returns the lag with highest absolute correlation.
    """
    sig = lag_results[lag_results['significant']]

    if sig.empty:
        return lag_results.loc[lag_results['correlation'].abs().idxmax()]

    return sig.loc[sig['correlation'].abs().idxmax()]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_overlay(sales_series, trends_df, category, best_lag):
    """
    Overlays normalized search volume and transaction volume on the
    same axis to visually demonstrate the leading indicator relationship.

    Both series are normalized to 0-100 scale so they are visually
    comparable despite having different absolute magnitudes.

    The search series is shifted forward by best_lag weeks so the
    alignment of the two series is visually apparent.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    slug = category.lower().replace('-', '_').replace('/', '_')

    # Normalize sales to 0-100
    sales = sales_series.set_index('ds')['y']
    sales_norm = (sales - sales.min()) / (sales.max() - sales.min()) * 100

    # Normalize search composite
    search = trends_df['composite'].resample('W').mean()
    search_norm = (search - search.min()) / (search.max() - search.min()) * 100

    # Align on common dates
    common = sales_norm.index.intersection(search_norm.index)
    sales_plot  = sales_norm.loc[common]
    search_plot = search_norm.loc[common].shift(best_lag['lag_weeks'])

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(sales_plot.index, sales_plot.values,
            color='steelblue', linewidth=1.5,
            label='H&M Transaction Volume (normalized)')

    ax.plot(search_plot.index, search_plot.values,
            color='darkorange', linewidth=1.5, linestyle='--',
            label=(f"Google Search Interest — shifted {best_lag['lag_weeks']}wk forward "
                   f"(r={best_lag['correlation']}, p={best_lag['pvalue']})"))

    ax.set_title(f'{category} — Search Interest vs Transaction Volume\n'
                 f"Best lag: {best_lag['lag_weeks']} weeks | "
                 f"r = {best_lag['correlation']} | "
                 f"p = {best_lag['pvalue']}",
                 fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Index (0-100)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, f'trends_overlay_{slug}.png'), dpi=150)
    plt.show()
    plt.close()


def plot_lag_correlations(lag_results, category):
    """
    Bar chart showing correlation at each lag value.
    Significant lags (p < 0.05) are highlighted in orange.
    Non-significant lags are shown in gray.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    slug = category.lower().replace('-', '_').replace('/', '_')

    colors = ['darkorange' if sig else 'lightgray'
              for sig in lag_results['significant']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(lag_results['lag_weeks'], lag_results['correlation'],
                  color=colors, edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.5,  color='green', linewidth=0.8, linestyle='--',
               alpha=0.5, label='r = 0.5 (moderate)')
    ax.axhline(y=0.7,  color='green', linewidth=0.8, linestyle='-',
               alpha=0.5, label='r = 0.7 (strong)')

    ax.set_title(f'{category} — Cross-Correlation by Lag\n'
                 f'(Orange = statistically significant, p < 0.05)',
                 fontsize=12)
    ax.set_xlabel('Lag (weeks) — search leads sales by this many weeks')
    ax.set_ylabel('Pearson Correlation (r)')
    ax.set_xticks(lag_results['lag_weeks'])
    ax.legend(fontsize=9)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, f'lag_correlation_{slug}.png'), dpi=150)
    plt.show()
    plt.close()


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary(all_lag_results):
    """
    Prints a clean summary table of leading indicator findings.
    """
    print(f"\n{'='*60}")
    print("GOOGLE TRENDS LEADING INDICATOR SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<12} {'Best Lag':>10} {'Correlation':>12} {'P-value':>10} {'Signal':>10}")
    print("-" * 58)

    for category, result in all_lag_results.items():
        best = result['best_lag']
        sig  = 'STRONG' if best['correlation'] >= 0.7 else \
               'MODERATE' if best['correlation'] >= 0.5 else \
               'WEAK'
        print(f"{category:<12} {best['lag_weeks']:>9}wk "
              f"{best['correlation']:>12.4f} "
              f"{best['pvalue']:>10.4f} "
              f"{sig:>10}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all(weekly_series):
    """
    Runs the full Google Trends analysis pipeline.

    1. Pulls search data for all four categories
    2. Runs cross-correlation at lags 0-8 weeks
    3. Identifies best lag per category
    4. Plots overlay and lag correlation charts
    5. Prints summary table

    Parameters
    ----------
    weekly_series : dict from load_data.build_weekly_series()

    Returns
    -------
    dict of {category: {'lag_results': df, 'best_lag': row}}
    """
    print("Pulling Google Trends data...")
    all_trends = get_all_trends()

    all_results = {}

    for category, series in weekly_series.items():
        if category not in all_trends:
            print(f"Skipping {category} — no trends data")
            continue

        print(f"\nAnalyzing {category}...")
        trends_df  = all_trends[category]
        lag_results = cross_correlate(series, trends_df)
        best_lag    = find_best_lag(lag_results)

        print(f"  Best lag: {best_lag['lag_weeks']} weeks | "
              f"r = {best_lag['correlation']} | "
              f"p = {best_lag['pvalue']}")

        plot_overlay(series, trends_df, category, best_lag)
        plot_lag_correlations(lag_results, category)

        all_results[category] = {
            'lag_results': lag_results,
            'best_lag':    best_lag
        }

    print_summary(all_results)
    return all_results


if __name__ == '__main__':
    """
    Usage:
        python python/trends.py

    Requires weekly series from load_data.py.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_data import load_merged, build_weekly_series

    df            = load_merged()
    weekly_series = build_weekly_series(df)
    results       = run_all(weekly_series)