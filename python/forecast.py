"""
forecast.py
===========
Builds Prophet demand forecasts, baseline models, and evaluation
metrics for each H&M product category.

THE FORECASTING PROBLEM IN FASHION
------------------------------------
Fashion demand forecasting is fundamentally different from general
retail. The core challenges:

1. LEAD TIME PARADOX
   Fashion brands must commit to production quantities 6-12 months
   before sale. But the cultural signals that drive demand may not
   emerge until weeks before launch. This model addresses the
   operational planning window — forecasting 12 weeks ahead, which
   aligns with reorder and replenishment cycles rather than initial
   production commitments.

2. STYLE PERISHABILITY
   Products lose economic value not through physical degradation but
   through shifting aesthetic norms. This means demand curves in
   fashion are not symmetric — a product can spike hard and drop off
   a cliff, unlike consumer staples which mean-revert predictably.
   Prophet's piecewise linear trend with changepoint detection handles
   this better than ARIMA-based approaches.

3. SEASONALITY IS STRUCTURAL
   Unlike CPG/grocery where seasonality is mild, fashion seasonality
   is a core business driver. A Sweater forecast that misses the
   October peak by 2 weeks causes real inventory and markdown costs.
   Prophet's Fourier-series seasonality decomposition is designed
   exactly for this — it separates the trend from seasonal cycles
   so planners can see both independently.

THE PROPHET MODEL
-----------------
Prophet decomposes the time series into additive components:

    y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)

TREND COMPONENT
    Prophet fits a piecewise linear trend — it detects points where
    the trend direction changes (changepoints) and fits separate line
    segments between them. This is appropriate for fashion where demand
    trends can shift abruptly due to style cycles.

    The changepoint_prior_scale parameter (set to 0.05) controls
    trend flexibility:
        Low value  (0.05) -> smoother trend, less sensitive to noise
        High value (0.5)  -> more flexible, risks overfitting short spikes

    We use 0.05 because fashion weekly data has high week-to-week noise
    (promotional spikes, weather anomalies) that we don't want the trend
    to chase.

SEASONALITY COMPONENT
    Prophet models seasonality using Fourier series — combinations of
    sine and cosine waves at different frequencies that sum to approximate
    any repeating pattern:

        s(t) = sum of [a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P)]

    where P is the period (365.25 days for yearly seasonality) and n
    determines the number of Fourier terms (complexity of the pattern).

    For fashion, yearly seasonality is the dominant signal:
        Sweater  -> peaks October, troughs June
        Swimwear -> peaks June, troughs November
        T-shirt  -> peaks May-July, mild trough in winter
        Trousers -> mild fall peak, relatively stable

    Weekly seasonality is turned OFF because H&M's weekly patterns
    reflect store traffic and paycheck cycles, not fashion demand cycles.
    Including them would add noise without meaningful signal for planning.

UNCERTAINTY INTERVALS
    Prophet outputs yhat_lower and yhat_upper — an 80% prediction
    interval meaning there is an 80% probability actual sales fall
    within this band. In fashion planning, the upper band informs
    safety stock calculations, the lower band informs minimum order
    commitments. Wider bands signal less forecastable categories.

COVID NOTE
    A demand shock is visible across all categories in early 2020
    (March-April). This represents a structural break that no model
    trained on prior data could predict. It is noted as a limitation
    and demonstrates exactly why the fashion industry's push toward
    real-time demand sensing (Zara's daily feedback loops, H&M's
    Google Cloud integration) is critical — historical models break
    during unprecedented events.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet

warnings.filterwarnings('ignore')

PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')


# ── Baseline Model ────────────────────────────────────────────────────────────

def build_baseline(series, window=12):
    """
    Builds a rolling average baseline forecast.

    WHAT IS A BASELINE
    ------------------
    A baseline is the simplest reasonable forecast — the performance
    floor. If a sophisticated model cannot beat a simple heuristic,
    the complexity is not justified. Without a baseline there is no
    way to prove the Prophet model is adding value over what a
    spreadsheet analyst could produce in minutes.

    THE 3-MONTH ROLLING AVERAGE
    ---------------------------
    Predicts next week's demand as the mean of the prior 12 weeks
    (3 months of weekly data):

        baseline(t) = (1/12) * sum(y(t-1) + y(t-2) + ... + y(t-12))

    This is what a non-technical merchandiser would actually do:
    "What did we sell last quarter on average?" It is a real business
    heuristic used in fashion planning teams globally.

    WHY NOT SEASONAL NAIVE
    ----------------------
    Seasonal Naive (predict = same week last year) is a strong baseline
    for core items. However it is a weak baseline for trend-driven fashion
    items because the same item often did not exist last year. For H&M
    categories at the product_type level, seasonal naive is appropriate
    since Trousers/Sweaters are perennial categories. We use rolling
    average instead because it is more conservative and easier to explain
    to a non-technical audience.

    Parameters
    ----------
    series : pd.DataFrame with ds and y columns
    window : int, number of weeks to average (default 12 = 3 months)

    Returns
    -------
    pd.Series of baseline predictions, aligned to series index.
    NaN for the first (window) rows where insufficient history exists.
    """
    return series['y'].rolling(window=window, min_periods=window).mean().shift(1)


# ── Category-Specific Tuning Parameters ──────────────────────────────────────

# Default parameters work well for Trousers, T-shirt, and Swimwear.
# Sweater required tuning due to its uniquely sharp October demand spike.
#
# SWEATER TUNING — conducted March 2026
# --------------------------------------
# Problem: default params (fourier_order=10, cps=0.05, n_cp=25) produced
# WMAPE=21.3% for Sweater — the worst of all four categories.
#
# Root cause: the default Fourier order of 10 generates a seasonality
# curve that is too smooth to capture Sweater's sharp October spike.
# The curve rounds off the peak, underestimating demand at the critical
# pre-winter inventory commitment point.
#
# Tuning approach: grid search across 8 parameter combinations testing:
#   changepoint_prior_scale : [0.05, 0.10, 0.20, 0.30]
#   fourier_order           : [10, 15, 20]
#   n_changepoints          : [25, 35]
#
# Key finding: fourier_order drove the improvement, not changepoint
# flexibility. Doubling Fourier terms from 10 to 20 allowed the model
# to represent the sharp peak shape accurately. changepoint_prior_scale
# had minimal impact (22.26% -> 22.15% across its full range at
# fourier=10), while fourier_order alone drove the major improvement
# (22.26% -> 16.77% at optimal settings).
#
# Winner: cps=0.1, fourier_order=20, n_changepoints=35 -> WMAPE=16.77%
# Improvement: 21.3% -> 16.77% (21.3% reduction in forecast error)
#
# FASHION INDUSTRY RELEVANCE
# ---------------------------
# This tuning result reflects a real operational insight: knitwear and
# outerwear categories have the sharpest seasonal demand curves in
# fashion retail. A generic forecasting model that treats all categories
# identically will systematically underforecast peak knitwear demand,
# leading to stockouts in October — the highest-margin selling period.
# Category-specific model tuning is standard practice at retailers like
# H&M and Zara who maintain separate model configurations per
# product group.

CATEGORY_PARAMS = {
    'Trousers': {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality':      10,
        'n_changepoints':          25,
    },
    'T-shirt': {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality':      10,
        'n_changepoints':          25,
    },
    'Sweater': {
        # Tuned: fourier_order=20 captures sharp October spike
        # WMAPE improved from 21.3% to 16.77%
        'changepoint_prior_scale': 0.10,
        'yearly_seasonality':      20,
        'n_changepoints':          35,
    },
    'Swimwear': {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality':      10,
        'n_changepoints':          25,
    },
}


# ── Prophet Forecast ──────────────────────────────────────────────────────────

def run_forecast(series, periods=12, category=None):
    """
    Fits a Prophet model to a weekly time series and generates
    a 12-week forward forecast.

    Uses category-specific parameters where available (see CATEGORY_PARAMS).
    Falls back to conservative defaults for any category not in the dict.

    PARAMETER DECISIONS
    -------------------
    yearly_seasonality : int (Fourier order) or True/False
        Controls the complexity of the seasonal curve. Higher values
        allow sharper peaks but risk overfitting. Sweater uses 20
        (tuned), all others use 10 (default).

    changepoint_prior_scale : float
        Controls trend flexibility. Lower = smoother trend.
        0.05 is conservative and appropriate for fashion weekly data
        which has high week-to-week noise from promotions and weather.

    weekly_seasonality : False
        Turned off. H&M weekly patterns reflect store traffic and
        paycheck cycles, not fashion demand signals. Including them
        adds noise without meaningful planning signal.

    interval_width : 0.80
        80% prediction interval. In fashion planning, yhat_upper
        informs safety stock calculations and yhat_lower informs
        minimum order commitments.

    Parameters
    ----------
    series   : pd.DataFrame with ds (datetime) and y (int) columns
    periods  : int, number of weeks to forecast ahead (default 12)
    category : str, category name to look up tuned params (optional)

    Returns
    -------
    model    : fitted Prophet model object
    forecast : pd.DataFrame with ds, yhat, yhat_lower, yhat_upper
               and component columns (trend, yearly, etc.)
    """
    # Use tuned params if available, else conservative defaults
    params = CATEGORY_PARAMS.get(category, {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality':      10,
        'n_changepoints':          25,
    })

    model = Prophet(
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=params['changepoint_prior_scale'],
        n_changepoints=params['n_changepoints'],
        interval_width=0.80
    )

    model.fit(series)

    future   = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)

    return model, forecast


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def evaluate(series, forecast, baseline):
    """
    Computes MAE, MAPE, and WMAPE for both the Prophet model and
    the rolling average baseline. Returns a summary dict.

    We evaluate on the in-sample period only (where we have actual
    values to compare against). Out-of-sample evaluation would require
    a held-out test set — noted as a future improvement.

    MAE — Mean Absolute Error
    -------------------------
    The average number of transactions the forecast is off per week.

        MAE = (1/n) * sum(|actual(t) - predicted(t)|)

    Intuition: if MAE = 400 for Trousers, the forecast is wrong by
    400 transactions per week on average. Useful for operational
    planning (how much safety stock to hold) but not comparable
    across categories with different volumes.

    MAPE — Mean Absolute Percentage Error
    --------------------------------------
    Percentage error normalized by actual volume:

        MAPE = (1/n) * sum(|actual(t) - predicted(t)| / actual(t)) * 100

    Allows cross-category comparison. An 8% MAPE means the forecast
    is off by 8% of actual sales on average regardless of volume.

    LIMITATION: undefined when actual(t) = 0 (division by zero).
    Fashion categories have zero-sales weeks for niche items. For our
    four high-volume categories this is not an issue (0 zero weeks
    confirmed in EDA) but noted as a limitation for future expansion.

    MAPE also has an underforecasting bias: equal absolute errors
    above vs below actual produce asymmetric percentage errors. A
    model optimized purely on MAPE will tend to underforecast, which
    in fashion leads to stockouts — more costly than overstock.

    WMAPE — Weighted Mean Absolute Percentage Error
    ------------------------------------------------
    The industry standard metric for fashion retail. Weights errors
    by actual sales volume so that high-volume weeks drive the score
    more than low-volume weeks:

        WMAPE = sum(|actual(t) - predicted(t)|) / sum(actual(t)) * 100

    Key advantages over MAPE:
    1. No division by zero — divides by sum of actuals, not individual
    2. No underforecasting bias — symmetric treatment of over/under
    3. Financially aligned — errors in high-revenue weeks matter more

    This is the PRIMARY metric for this project. All model comparison
    conclusions are drawn from WMAPE.

    BEATS BASELINE
    --------------
    Boolean flag indicating whether the Prophet model's WMAPE is lower
    than the baseline's WMAPE. This is the business validation:
    does the sophistication of Prophet justify the complexity over
    a simple rolling average?

    Parameters
    ----------
    series   : pd.DataFrame with ds and y columns (actual values)
    forecast : pd.DataFrame output from Prophet (contains yhat)
    baseline : pd.Series of rolling average predictions

    Returns
    -------
    dict with keys: mae, mape, wmape, baseline_mae, baseline_mape,
                    baseline_wmape, beats_baseline
    """
    # Align forecast with actual values (in-sample only)
    actual    = series['y'].values
    predicted = forecast.loc[forecast['ds'].isin(series['ds']), 'yhat'].values
    base      = baseline.values

    # Trim to same length and remove NaN baseline rows
    valid     = ~np.isnan(base)
    actual    = actual[valid]
    predicted = predicted[valid]
    base      = base[valid]

    # ── Prophet metrics ───────────────────────────────────────────

    mae   = np.mean(np.abs(actual - predicted))

    # MAPE: guard against zero actuals
    nonzero = actual != 0
    mape    = np.mean(np.abs(actual[nonzero] - predicted[nonzero])
                      / actual[nonzero]) * 100

    # WMAPE: sum of absolute errors / sum of actuals
    wmape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

    # ── Baseline metrics ──────────────────────────────────────────

    b_mae   = np.mean(np.abs(actual - base))

    b_nonzero = actual != 0
    b_mape    = np.mean(np.abs(actual[b_nonzero] - base[b_nonzero])
                        / actual[b_nonzero]) * 100

    b_wmape = np.sum(np.abs(actual - base)) / np.sum(actual) * 100

    return {
        'prophet_mae':      round(mae,   1),
        'prophet_mape':     round(mape,  2),
        'prophet_wmape':    round(wmape, 2),
        'baseline_mae':     round(b_mae,   1),
        'baseline_mape':    round(b_mape,  2),
        'baseline_wmape':   round(b_wmape, 2),
        'beats_baseline':   wmape < b_wmape
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_forecast(series, model, forecast, baseline, category, metrics):
    """
    Produces two charts per category:

    Chart 1 — Forecast vs Actual vs Baseline
        Shows the full actual series (blue), Prophet forecast (red)
        with uncertainty band, baseline (gray dashed), and a vertical
        line marking where the historical data ends and the 12-week
        forward forecast begins.

    Chart 2 — Prophet Components
        Decomposes the forecast into trend and yearly seasonality.
        This is the most analytically interesting output — the
        seasonality plot shows the within-year demand pattern for
        the category, which is exactly what a merchandising team
        uses to time buys and replenishment orders.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    slug = category.lower().replace('-', '_').replace('/', '_')

    # ── Chart 1: Forecast ─────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual
    ax.plot(series['ds'], series['y'],
            color='steelblue', linewidth=1.5,
            label='Actual', zorder=3)

    # Baseline
    ax.plot(series['ds'], baseline,
            color='gray', linewidth=1.2, linestyle='--',
            label=f"Baseline (12-wk rolling avg) | WMAPE: {metrics['baseline_wmape']}%",
            zorder=2)

    # Prophet forecast + uncertainty
    ax.plot(forecast['ds'], forecast['yhat'],
            color='tomato', linewidth=1.5,
            label=f"Prophet Forecast | WMAPE: {metrics['prophet_wmape']}%",
            zorder=3)

    ax.fill_between(forecast['ds'],
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    color='tomato', alpha=0.15,
                    label='80% Uncertainty Interval')

    # Forecast start line
    forecast_start = series['ds'].max()
    ax.axvline(x=forecast_start,
               color='black', linestyle=':', linewidth=1.2,
               label='Forecast Start')

    # COVID annotation
    covid_start = pd.Timestamp('2020-03-01')
    ax.axvspan(covid_start, pd.Timestamp('2020-05-01'),
               alpha=0.08, color='orange',
               label='COVID demand shock')

    ax.set_title(f'H&M {category} — Demand Forecast vs Baseline',
                 fontsize=13, pad=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Transactions')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, f'forecast_{slug}.png'), dpi=150)
    plt.show()
    plt.close()

    # ── Chart 2: Components ───────────────────────────────────────

    fig2 = model.plot_components(forecast)
    fig2.suptitle(f'{category} — Trend and Seasonality Components',
                  fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, f'components_{slug}.png'), dpi=150)
    plt.show()
    plt.close()


def plot_metrics_comparison(results):
    """
    Bar chart comparing WMAPE across all categories for Prophet vs Baseline.
    The primary visual for the findings section.
    """
    categories = list(results.keys())
    prophet_wmape  = [results[c]['prophet_wmape']  for c in categories]
    baseline_wmape = [results[c]['baseline_wmape'] for c in categories]

    x     = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_wmape, width,
                   label='Baseline (Rolling Avg)', color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, prophet_wmape, width,
                   label='Prophet Forecast', color='tomato', alpha=0.8)

    ax.set_title('Forecast Accuracy: Prophet vs Baseline\n(Lower WMAPE = Better)',
                 fontsize=13)
    ax.set_ylabel('WMAPE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'metrics_comparison.png'), dpi=150)
    plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all(weekly_series):
    """
    Runs the full forecasting pipeline for all categories.
    Fits Prophet, builds baseline, evaluates metrics, plots results.

    Parameters
    ----------
    weekly_series : dict from load_data.build_weekly_series()

    Returns
    -------
    results : dict of {category: metrics_dict}
    """
    results = {}

    for name, series in weekly_series.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        # Baseline
        baseline = build_baseline(series)

        # Prophet — uses category-specific tuned params
        print(f"  Fitting Prophet...")
        model, forecast = run_forecast(series, category=name)

        # Evaluate
        metrics = evaluate(series, forecast, baseline)
        results[name] = metrics

        print(f"  Prophet WMAPE : {metrics['prophet_wmape']}%")
        print(f"  Baseline WMAPE: {metrics['baseline_wmape']}%")
        print(f"  Beats baseline: {metrics['beats_baseline']}")

        # Plot
        plot_forecast(series, model, forecast, baseline, name, metrics)

    # Summary table
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Category':<12} {'P.WMAPE':>8} {'B.WMAPE':>8} {'Beats?':>8}")
    print("-" * 40)
    for name, m in results.items():
        beat = 'YES' if m['beats_baseline'] else 'NO'
        print(f"{name:<12} {m['prophet_wmape']:>7}% {m['baseline_wmape']:>7}% {beat:>8}")

    # Comparison chart
    plot_metrics_comparison(results)

    return results


if __name__ == '__main__':
    """
    Usage:
        python python/forecast.py

    Requires data/processed/ CSVs built by load_data.py.
    Or import and pass weekly_series dict directly from a notebook.
    """
    from load_data import load_merged, build_weekly_series

    df            = load_merged()
    weekly_series = build_weekly_series(df)
    results       = run_all(weekly_series)