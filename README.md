# Fashion Demand Forecasting

A demand forecasting model built on 31.7 million H&M transactions (2018–2020), 
testing whether Google Trends search volume acts as a leading indicator for 
fashion purchase behavior across four product categories.

**Live dashboard → [View here](https://hasnaoh.github.io/fashion-trend-forecasting/)**

---

## What This Project Does

Uses Facebook Prophet to forecast weekly demand for four H&M product categories 
(Trousers, T-shirt, Sweater, Swimwear) and tests whether US Google Trends search 
volume predicts transaction volume 1–4 weeks in advance.

---

## Key Findings

1. **Prophet beats a 3-month rolling average baseline in all four categories.** 
   Improvement scales with seasonal intensity — Swimwear improved 65%, Sweater 56%.

2. **Google search predicts swimwear purchases 1 week in advance** (r = 0.88, 
   p < 0.001). A demand planner monitoring search trends has a meaningful window 
   to adjust inventory before the spike hits.

3. **Search signal strength is category-dependent.** Trousers shows no significant 
   relationship (r = 0.11, p = 0.27) because it aggregates subcategories with 
   opposing seasonal patterns. Subcategory-level analysis would likely yield 
   stronger signals.

---

## Results

| Category | Prophet WMAPE | Baseline WMAPE | Improvement | Trends Signal |
|----------|--------------|----------------|-------------|---------------|
| Trousers | 13.28%       | 16.0%          | 17%         | Weak          |
| T-shirt  | 12.93%       | 28.52%         | 55%         | Moderate      |
| Sweater  | 16.77%*      | 48.29%         | 65%         | Moderate      |
| Swimwear | 19.71%       | 56.2%          | 65%         | Strong        |

*Sweater model tuned — Fourier order increased from 10 to 20 after grid search.

---

## Tech Stack

- **Python** — data pipeline, modeling, analysis
- **Facebook Prophet** — time series forecasting
- **pytrends** — Google Trends API
- **Pandas, NumPy, SciPy** — data manipulation and statistics
- **Matplotlib** — visualization
- **Kaggle API** — data download
- **HTML / CSS / JavaScript** — interactive dashboard

---

## Project Structure
