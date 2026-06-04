"""
BESS Dispatch Optimizer
Reads DA prices from PostgreSQL, optimizes daily battery dispatch using DP,
writes results to CSV and prints daily P&L summary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, get_engine

# -----------------------------
# Config
# -----------------------------
BATTERY = {
    "capacity_mwh": 100,
    "efficiency": 0.9,
    "max_cycles_per_day": 2,
    "soc_min_pct": 0.10,
    "soc_max_pct": 0.90,
}

def load_prices(engine):
    query = """
    SELECT
        p.datetime,
        p.price_eur_mwh,
        COALESCE(g.wind_generation, 0.0) as wind_generation,
        COALESCE(g.solar_generation, 0.0) as solar_generation
    FROM prices p
    LEFT JOIN (
        SELECT
            datetime,
            SUM(CASE WHEN production_technology IN ('Wind Onshore', 'Wind Offshore') THEN actual_mwh ELSE 0 END) as wind_generation,
            SUM(CASE WHEN production_technology = 'Solar' THEN actual_mwh ELSE 0 END) as solar_generation
        FROM generation
        GROUP BY datetime
    ) g ON p.datetime = g.datetime
    ORDER BY p.datetime
    """
    df = pd.read_sql(query, engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# -----------------------------
# DP Optimizer
# -----------------------------
def optimize_day(prices, capacity, efficiency, max_cycles, soc_min, soc_max, start_soc):
    """
    SOC-constrained DP optimizer.
    State: (soc_level, charges_done). Charges from start_soc=0 to soc_min are
    initial fill-up and do not count toward max_cycles.
    Returns (profit, schedule) where schedule is a list of (index, action) tuples.
    """
    T = len(prices)
    if T == 0:
        return 0.0, []

    usable = soc_max - soc_min

    # parent[t+1][state] = (prev_state, action) for path reconstruction
    parent = [{} for _ in range(T + 1)]
    dp = {(start_soc, 0): 0.0}

    for t, price in enumerate(prices):
        new_dp = {}

        def update(new_s, prev_s, val, action, _t=t):
            if new_s not in new_dp or val > new_dp[new_s]:
                new_dp[new_s] = val
                parent[_t + 1][new_s] = (prev_s, action)

        for (soc, c), profit in dp.items():
            update((soc, c), (soc, c), profit, "Hold")

            # Initial fill: 0 → soc_min, does not count as a charge cycle
            if soc == 0:
                cost = soc_min * price / efficiency
                update((soc_min, c), (soc, c), profit - cost, "Charge")

            # Regular charge: soc_min → soc_max
            if soc == soc_min and c < max_cycles:
                cost = usable * price / efficiency
                update((soc_max, c + 1), (soc, c), profit - cost, "Charge")

            # Discharge: soc_max → soc_min
            if soc == soc_max:
                revenue = usable * price * efficiency
                update((soc_min, c), (soc, c), profit + revenue, "Discharge")

        dp = new_dp

    best_profit = 0.0
    best_state = None
    for state, profit in dp.items():
        if profit > best_profit:
            best_profit = profit
            best_state = state

    if best_state is None:
        return 0.0, []

    schedule = []
    curr = best_state
    for t in range(T - 1, -1, -1):
        if curr not in parent[t + 1]:
            break
        prev, action = parent[t + 1][curr]
        if action != "Hold":
            schedule.append((t, action))
        curr = prev

    schedule.reverse()
    return best_profit, schedule

# -----------------------------
# Run backtest
# -----------------------------
def run_backtest(df, battery):
    """Runs the optimizer on each day and returns dispatch log + daily P&L."""
    soc_min = battery["capacity_mwh"] * battery["soc_min_pct"]
    soc_max = battery["capacity_mwh"] * battery["soc_max_pct"]
    efficiency = battery["efficiency"]

    all_dispatches = []
    day_start_soc = 0.0  # battery starts completely empty

    for date, group in df.groupby(df["datetime"].dt.date):
        day_prices = group["price_eur_mwh"].tolist()
        day_datetimes = group["datetime"].tolist()

        _, schedule = optimize_day(
            day_prices,
            battery["capacity_mwh"],
            efficiency,
            battery["max_cycles_per_day"],
            soc_min,
            soc_max,
            day_start_soc,
        )

        scheduled = {idx: action for idx, action in schedule}
        soc = day_start_soc

        for i, (dt, price) in enumerate(zip(day_datetimes, day_prices)):
            action = scheduled.get(i, "Hold")
            soc_before = soc

            if action == "Charge":
                soc = soc_min if soc_before == 0 else soc_max
                mwh = soc - soc_before
                value = -mwh * price / efficiency
            elif action == "Discharge":
                soc = soc_min
                mwh = soc_before - soc
                value = mwh * price * efficiency
            else:
                mwh = 0.0
                value = 0.0

            all_dispatches.append({
                "date": date,
                "datetime": dt,
                "action": action,
                "price_eur_mwh": round(price, 2),
                "mwh": round(mwh, 2),
                "value_eur": round(value, 2),
                "soc_before_mwh": round(soc_before, 2),
                "soc_after_mwh": round(soc, 2),
            })

        day_start_soc = soc  # carry end-of-day SOC into next day

    res_df = pd.DataFrame(all_dispatches)
    if "wind_generation" in df.columns and "solar_generation" in df.columns:
        res_df = res_df.merge(df[["datetime", "wind_generation", "solar_generation"]], on="datetime", how="left")
    return res_df

# -----------------------------
# Visualizations
# -----------------------------
def plot_dispatch(results, total_profit):
    """Full-period overview: daily P&L bars + cumulative profit."""
    daily = results.groupby("date")["value_eur"].sum().reset_index()
    daily.columns = ["date", "daily_profit"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    colors = ["green" if p > 0 else "red" for p in daily["daily_profit"]]
    ax1.bar(range(len(daily)), daily["daily_profit"], color=colors, width=0.8)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Profit [€]")
    ax1.set_title("Daily BESS Profit")
    ax1.axhline(0, color="black", linewidth=0.5)

    ax2.plot(results["datetime"], results["cumulative_profit"], color="steelblue", linewidth=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cumulative Profit [€]")
    ax2.set_title(f"Cumulative BESS Profit — Total: €{total_profit:,.2f}")
    ax2.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}\\bess_performance.png")
    plt.show()


def plot_dispatch_detail(results):
    """Three-panel chart for the full period: price + dispatch markers, SOC, interval P&L."""
    charges = results[results["action"] == "Charge"]
    discharges = results[results["action"] == "Discharge"]
    active = results[results["action"] != "Hold"]
    soc_min_val = active["soc_after_mwh"].min() if not active.empty else 0
    soc_max_val = results["soc_after_mwh"].max()

    start = results["datetime"].iloc[0]
    end = results["datetime"].iloc[-1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"BESS Dispatch — {start.date()} to {end.date()}", fontsize=13)

    # --- Panel 1: price + action markers ---
    ax1.plot(results["datetime"], results["price_eur_mwh"],
             color="dimgray", linewidth=0.6, label="Price")
    ax1.scatter(charges["datetime"], charges["price_eur_mwh"],
                color="royalblue", zorder=5, s=20, label="Charge")
    ax1.scatter(discharges["datetime"], discharges["price_eur_mwh"],
                color="tomato", zorder=5, s=20, label="Discharge")
    ax1.set_ylabel("Price [€/MWh]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    # --- Panel 2: SOC ---
    ax2.fill_between(results["datetime"], results["soc_after_mwh"], alpha=0.25, color="steelblue")
    ax2.plot(results["datetime"], results["soc_after_mwh"], color="steelblue", linewidth=0.8)
    ax2.axhline(soc_min_val, color="orange", linewidth=0.8, linestyle="--",
                label=f"SOC min ({soc_min_val:.0f} MWh)")
    ax2.axhline(soc_max_val, color="green", linewidth=0.8, linestyle="--",
                label=f"SOC max ({soc_max_val:.0f} MWh)")
    ax2.set_ylabel("SOC [MWh]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    # --- Panel 3: interval P&L ---
    pnl_colors = ["green" if v > 0 else ("red" if v < 0 else "lightgray")
                  for v in results["value_eur"]]
    if len(results) > 1:
        interval = (results["datetime"].iloc[1] - results["datetime"].iloc[0]).total_seconds() / 86400
    else:
        interval = 1 / 24
    ax3.bar(results["datetime"], results["value_eur"], color=pnl_colors, width=interval * 0.9)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Interval P&L [€]")
    ax3.set_xlabel("Time")
    ax3.grid(axis="y", linewidth=0.4, alpha=0.5)

    # --- New Figure: Wind Generation vs DA Price, correlation --- #
    plt.figure(figsize=(12,6))
    plt.scatter(results["wind_generation"],results["price_eur_mwh"])
    plt.xlabel("Wind Generation [MWh]")
    plt.ylabel("Price [€/MWh]")
    plt.title("Wind Generation vs DA Price")
    linear_fit = np.polyfit(results["wind_generation"],results["price_eur_mwh"], 1)
    plt.plot(results["wind_generation"], linear_fit[0]*results["wind_generation"] + linear_fit[1], color="red")
    plt.grid(True)
    correlation = np.corrcoef(results["wind_generation"],results["price_eur_mwh"])[0, 1]
    plt.text(x=0.05, y=0.95, s=f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig(f"{OUTPUT_DIR}\\wind_correlation.png")

    # --- New Figure: Wind Generation & Solar vs DA Price, correlation --- #
    plt.figure(figsize=(12,6))
    plt.scatter(results["wind_generation"] + results["solar_generation"],results["price_eur_mwh"])
    plt.xlabel("Wind + Solar Generation [MWh]")
    plt.ylabel("Price [€/MWh]")
    plt.title("Wind + Solar Generation vs DA Price")
    linear_fit = np.polyfit(results["wind_generation"] + results["solar_generation"],results["price_eur_mwh"], 1)
    plt.plot(results["wind_generation"] + results["solar_generation"], linear_fit[0]*(results["wind_generation"] + results["solar_generation"]) + linear_fit[1], color="red")
    plt.grid(True)
    correlation = np.corrcoef(results["wind_generation"] + results["solar_generation"],results["price_eur_mwh"])[0, 1]
    plt.text(x=0.05, y=0.95, s=f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig(f"{OUTPUT_DIR}\\wind_solar_correlation.png")

    plt.figure(figsize=(12,6))
    plt.scatter(results["wind_generation"], results["price_eur_mwh"], color = "blue")
    plt.scatter(results["solar_generation"], results["price_eur_mwh"], color = "orange")
    plt.xlabel("Generation [MWh]")
    plt.ylabel("Price [€/MWh]")
    plt.title("Wind & Solar Generation vs DA Price")
    plt.legend(["Wind", "Solar"])
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}\\wind_solar_correlation.png")

    plt.figure(figsize=(12,6))
    plt.scatter(results["wind_generation"], results["solar_generation"])
    plt.xlabel("Wind Generation [MWh]")
    plt.ylabel("Solar Generation [MWh]")
    plt.title("Wind & Solar Generation")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}\\wind_solar_generation.png")


    # --- Figure below wind: Solar Generation vs DA Price, correlation ---#
    plt.figure(figsize=(12,6))
    plt.scatter(results["solar_generation"],results["price_eur_mwh"])
    plt.xlabel("Solar Generation [MWh]")
    plt.ylabel("Price [€/MWh]")
    plt.title("Solar Generation vs DA Price")
    linear_fit = np.polyfit(results["solar_generation"],results["price_eur_mwh"], 1)
    plt.plot(results["solar_generation"], linear_fit[0]*results["solar_generation"] + linear_fit[1], color="red")
    plt.grid(True)
    correlation = np.corrcoef(results["solar_generation"],results["price_eur_mwh"])[0, 1]
    plt.text(x=0.05, y=0.95, s=f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig(f"{OUTPUT_DIR}\\solar_correlation.png")

    fig.savefig(f"{OUTPUT_DIR}\\dispatch_full_period.png")
    plt.show()

    # --- Twin-axis: price on left y-axis, SoC on right y-axis, shared ax-ais of time for last day --- #
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()
    ax1.plot(results["datetime"],results["price_eur_mwh"], color="blue", label="Price [€/MWh]")
    ax2.plot(results["datetime"],results["soc_after_mwh"], color="grey", label="SOC [MWh]", alpha = 0.4, linewidth = 0.4)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price [€/MWh]")
    ax2.set_ylabel("SOC [MWh]")
    plt.title("Price & SOC for the last day")
    fig.legend(loc="upper right")
    plt.show()



# -----------------------------
# Results & export
# -----------------------------
def summarize_and_export(results):
    """Prints summary, saves CSVs, and plots daily P&L."""
    if results.empty:
        print("No trades found.")
        return

    results["cumulative_profit"] = results["value_eur"].cumsum()

    daily = results.groupby("date")["value_eur"].sum().reset_index()
    daily.columns = ["date", "daily_profit"]

    total = daily["daily_profit"].sum()
    avg = daily["daily_profit"].mean()
    best = daily.loc[daily["daily_profit"].idxmax()]
    worst = daily.loc[daily["daily_profit"].idxmin()]

    print("\n===== BESS BACKTEST RESULTS =====")
    print(f"Period:         {daily['date'].min()} to {daily['date'].max()}")
    print(f"Days traded:    {len(daily)}")
    print(f"Total profit:   €{total:,.2f}")
    print(f"Avg daily:      €{avg:,.2f}")
    print(f"Best day:       {best['date']} (€{best['daily_profit']:,.2f})")
    print(f"Worst day:      {worst['date']} (€{worst['daily_profit']:,.2f})")
    print("=================================\n")

    # Save to CSV
    results.to_csv(f"{OUTPUT_DIR}\\dispatch_results.csv", index=False)
    daily.to_csv(f"{OUTPUT_DIR}\\daily_profit.csv", index=False)
    print(f"Saved: dispatch_results.csv, daily_profit.csv")

    plot_dispatch(results, total)
    plot_dispatch_detail(results)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    engine = get_engine()
    df = load_prices(engine)
    print(f"Loaded {len(df)} price rows ({df['datetime'].min()} to {df['datetime'].max()})")

    results = run_backtest(df, BATTERY)
    summarize_and_export(results)