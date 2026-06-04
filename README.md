# BESS Analytics Dashboard

A Streamlit dashboard for analysing German electricity market data and optimising Battery Energy Storage System (BESS) dispatch.

## What it does

- **Morning Report** — KPI cards (avg price, peak, spread, BESS P&L, wind/solar generation) for yesterday vs the day before, with price curve charts and Excel export.
- **BESS Dispatch** — Optimal charge/discharge schedule over the last 5 days, plotted against prices, wind, solar generation, and state of charge.
- **Price Analysis** — Price duration curve, yesterday vs day-before comparison, negative price periods, and monthly peak/off-peak averages.
- **Weekly Report** — Week-on-week BESS P&L comparison with daily detail and dispatch visualisation.
- **Wind Forecast** — Forecast vs actual for Wind Onshore and Offshore, with daily mean error charts and MAE summary.

The sidebar lets you adjust battery parameters (capacity, efficiency, cycles/day, SoC limits) and instantly see how they affect all dispatch and P&L calculations.

## Data sources

- Day-ahead electricity prices: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- Wind and solar generation: ENTSO-E Transparency Platform
- Data is stored in a local PostgreSQL database (`ENTSOE-E`)

## Tech stack

- Python 3.13
- Streamlit
- pandas, NumPy, matplotlib
- SQLAlchemy + psycopg2
- PostgreSQL

## Installation

```bash
pip install -r requirements.txt
```

## Running locally

```bash
python -m streamlit run dashboard.py
```

The app will open at `http://localhost:8501`.

Set the `ENTSOE_API_KEY` and `DB_PASSWORD` environment variables before running:

```bash
set ENTSOE_API_KEY=your_key_here
set DB_PASSWORD=your_db_password
```
