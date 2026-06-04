import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sqlalchemy import text
from config import get_engine
from Dispatcher import load_prices, run_backtest, optimize_day, BATTERY

st.set_page_config(page_title="BESS Dashboard", layout="wide")
st.title("BESS Analytics Dashboard")


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _prices():
    return load_prices(get_engine())


@st.cache_data(ttl=3600)
def _wind_forecast_df():
    engine = get_engine()
    q = text("""
        SELECT datetime, source, forecast_mwh, actual_mwh
        FROM generation
        WHERE source IN (
            'Wind Onshore Actual Aggregated',
            'Wind Offshore Actual Aggregated'
        )
        ORDER BY datetime, source;
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(ttl=3600)
def _dispatch_5d():
    df = _prices()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=5)
    last5 = df[df["datetime"] >= cutoff].copy()
    return last5, run_backtest(last5, BATTERY)


def _daily_detail(week_df, label):
    rows = []
    for date, group in week_df.groupby(week_df["datetime"].dt.date):
        profit, _ = optimize_day(group["price_eur_mwh"].tolist(), 100, 0.9, 2, 10, 90, 0)
        rows.append({
            "Week": label,
            "Date": date,
            "Avg Price [EUR/MWh]": round(group["price_eur_mwh"].mean(), 2),
            "Peak Price [EUR/MWh]": round(group["price_eur_mwh"].max(), 2),
            "Min Price [EUR/MWh]": round(group["price_eur_mwh"].min(), 2),
            "Spread [EUR/MWh]": round(group["price_eur_mwh"].max() - group["price_eur_mwh"].min(), 2),
            "BESS Optimal P&L [EUR]": round(profit, 2),
            "Negative Price Hours": int((group["price_eur_mwh"] < 0).sum()),
            "Wind Generation [MWh]": round(group["wind_generation"].sum(), 2),
            "Solar Generation [MWh]": round(group["solar_generation"].sum(), 2),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def _weekly_computation():
    df = _prices()
    max_date = pd.Timestamp(df["datetime"].max().date())
    days_since_friday = (max_date.weekday() - 4) % 7
    last_fri = max_date - pd.Timedelta(days=days_since_friday)
    last_mon = last_fri - pd.Timedelta(days=4)
    prev_fri = last_fri - pd.Timedelta(days=7)
    prev_mon = last_mon - pd.Timedelta(days=7)

    last_wk = df[
        (df["datetime"].dt.date >= last_mon.date()) &
        (df["datetime"].dt.date <= last_fri.date())
    ]
    prev_wk = df[
        (df["datetime"].dt.date >= prev_mon.date()) &
        (df["datetime"].dt.date <= prev_fri.date())
    ]

    last_label = f"Last Week ({last_mon.date()} - {last_fri.date()})"
    prev_label = f"Prev Week ({prev_mon.date()} - {prev_fri.date()})"

    last_daily = _daily_detail(last_wk, last_label)
    prev_daily = _daily_detail(prev_wk, prev_label)

    both = pd.concat([prev_wk, last_wk]).sort_values("datetime")
    last_dates = set(last_wk["datetime"].dt.date.unique())
    lw_c, lw_d, pw_c, pw_d = [], [], [], []
    for _, group in both.groupby(both["datetime"].dt.date):
        datetimes = group["datetime"].tolist()
        prices = group["price_eur_mwh"].tolist()
        date = group["datetime"].dt.date.iloc[0]
        is_last = date in last_dates
        _, schedule = optimize_day(prices, 100, 0.9, 2, 10, 90, 0)
        for idx, action in schedule:
            if action == "Charge":
                (lw_c if is_last else pw_c).append((datetimes[idx], prices[idx]))
            else:
                (lw_d if is_last else pw_d).append((datetimes[idx], prices[idx]))

    return (last_mon, last_fri, prev_mon, prev_fri,
            last_label, prev_label, last_daily, prev_daily, both,
            lw_c, lw_d, pw_c, pw_d)


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Morning Report", "BESS Dispatch", "Price Analysis", "Weekly Report", "Wind Forecast"
])


# ── Tab 1: Morning Report ────────────────────────────────────────────────────

with tab1:
    st.header("Morning Report")
    df = _prices()
    max_date = df["datetime"].max().date()

    yesterday = df[df["datetime"].dt.date == max_date - pd.Timedelta(days=1)]
    day_before = df[df["datetime"].dt.date == max_date - pd.Timedelta(days=2)]
    last7 = df[df["datetime"].dt.date >= max_date - pd.Timedelta(days=7)]

    y_avg    = yesterday["price_eur_mwh"].mean()
    y_peak   = yesterday["price_eur_mwh"].max()
    y_min    = yesterday["price_eur_mwh"].min()
    y_spread = y_peak - y_min
    db_avg   = day_before["price_eur_mwh"].mean()
    db_peak  = day_before["price_eur_mwh"].max()
    db_min   = day_before["price_eur_mwh"].min()
    db_spread = db_peak - db_min
    l7_avg   = last7["price_eur_mwh"].mean()

    y_wind   = yesterday["wind_generation"].sum()
    y_solar  = yesterday["solar_generation"].sum()
    db_wind  = day_before["wind_generation"].sum()
    db_solar = day_before["solar_generation"].sum()

    profit_y,  _ = optimize_day(yesterday["price_eur_mwh"].tolist(),  100, 0.9, 2, 10, 90, 0)
    profit_db, _ = optimize_day(day_before["price_eur_mwh"].tolist(), 100, 0.9, 2, 10, 90, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Price",        f"{y_avg:.2f} EUR/MWh",   f"{y_avg - db_avg:+.2f} vs prev day")
    c2.metric("Peak Price",       f"{y_peak:.2f} EUR/MWh",  f"{y_peak - db_peak:+.2f}")
    c3.metric("Min Price",        f"{y_min:.2f} EUR/MWh",   f"{y_min - db_min:+.2f}")
    c4.metric("Spread",           f"{y_spread:.2f} EUR/MWh", f"{y_spread - db_spread:+.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("BESS Optimal P&L", f"EUR {profit_y:,.2f}",   f"{profit_y - profit_db:+.2f} vs prev day")
    c6.metric("Wind Generation",  f"{y_wind:,.0f} MWh",     f"{y_wind - db_wind:+,.0f}")
    c7.metric("Solar Generation", f"{y_solar:,.0f} MWh",    f"{y_solar - db_solar:+,.0f}")
    c8.metric("7-Day Avg Price",  f"{l7_avg:.2f} EUR/MWh",  f"{y_avg - l7_avg:+.2f} vs avg")

    summary_df = pd.DataFrame({
        "Date":                [max_date - pd.Timedelta(days=1), max_date - pd.Timedelta(days=2)],
        "Avg Price [EUR/MWh]": [round(y_avg, 2),    round(db_avg, 2)],
        "Peak [EUR/MWh]":      [round(y_peak, 2),   round(db_peak, 2)],
        "Min [EUR/MWh]":       [round(y_min, 2),    round(db_min, 2)],
        "Spread [EUR/MWh]":    [round(y_spread, 2), round(db_spread, 2)],
        "BESS Profit [EUR]":   [round(profit_y, 2), round(profit_db, 2)],
        "Wind [MWh]":          [round(y_wind, 2),   round(db_wind, 2)],
        "Solar [MWh]":         [round(y_solar, 2),  round(db_solar, 2)],
    })
    st.dataframe(summary_df, use_container_width=True)

    buf = BytesIO()
    summary_df.to_excel(buf, index=False, engine="xlsxwriter")
    st.download_button("Download Morning Report", buf.getvalue(), "morning_report.xlsx")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yesterday["datetime"], yesterday["price_eur_mwh"], color="steelblue", label="Yesterday")
    ax.axhline(y_avg,  color="red",   linestyle="--", label=f"Avg: {y_avg:.2f}")
    ax.axhline(y_peak, color="green", linestyle="--", label=f"Peak: {y_peak:.2f}")
    ax.axhline(y_min,  color="gold",  linestyle="--", label=f"Min: {y_min:.2f}")
    ax.set_xlabel("Time"); ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title("Yesterday Price Curve"); ax.legend(); ax.grid(True)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(last7["datetime"], last7["price_eur_mwh"], color="blue", label="Last 7 Days")
    ax2.axhline(y_avg,  color="red",   linestyle="--", label=f"Yesterday avg: {y_avg:.2f}")
    ax2.axhline(l7_avg, color="green", linestyle="--", label=f"7-day avg: {l7_avg:.2f}")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Price [EUR/MWh]")
    ax2.set_title("Yesterday vs Last 7 Days Average"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ── Tab 2: BESS Dispatch ─────────────────────────────────────────────────────

with tab2:
    st.header("BESS Dispatch - Last 5 Days")
    last5, dispatch = _dispatch_5d()

    charges    = dispatch[dispatch["action"] == "Charge"]
    discharges = dispatch[dispatch["action"] == "Discharge"]
    avg_price  = last5["price_eur_mwh"].mean()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))

    ax1.plot(last5["datetime"], last5["price_eur_mwh"], color="grey", alpha=0.4, label="Price [EUR/MWh]")
    ax1.axhline(avg_price, color="grey", linestyle="--", label=f"Avg Price ({avg_price:.1f})")
    ax1.scatter(charges["datetime"],    charges["price_eur_mwh"],    color="green", marker="^", s=60, zorder=5, label="Charge")
    ax1.scatter(discharges["datetime"], discharges["price_eur_mwh"], color="red",   marker="v", s=60, zorder=5, label="Discharge")
    ax2.plot(last5["datetime"], last5["wind_generation"],  color="seagreen",  label="Wind Generation",  linewidth=0.8)
    ax2.plot(last5["datetime"], last5["solar_generation"], color="goldenrod", label="Solar Generation", linewidth=0.8)
    ax3.plot(dispatch["datetime"], dispatch["soc_after_mwh"], color="steelblue", linestyle=":", linewidth=1.2, label="SoC [MWh]")
    ax3.fill_between(dispatch["datetime"], dispatch["soc_after_mwh"], alpha=0.1, color="steelblue")

    ax1.set_xlabel("Time"); ax1.set_ylabel("Price [EUR/MWh]")
    ax2.set_ylabel("Generation [MWh]"); ax3.set_ylabel("SoC [MWh]")
    plt.title("Last 5 Days - Prices, Generation & BESS Dispatch")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    l3, lb3 = ax3.get_legend_handles_labels()
    ax1.legend(l1 + l2 + l3, lb1 + lb2 + lb3, fontsize=8, loc="upper left")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    daily = dispatch.groupby("date")["value_eur"].sum().reset_index()
    daily.columns = ["date", "daily_profit"]
    total = daily["daily_profit"].sum()
    colors = ["seagreen" if p >= 0 else "tomato" for p in daily["daily_profit"]]
    fig2, ax2b = plt.subplots(figsize=(10, 5))
    ax2b.bar(daily["date"].astype(str), daily["daily_profit"], color=colors)
    ax2b.axhline(0, color="black", linewidth=0.5)
    ax2b.set_xlabel("Date"); ax2b.set_ylabel("Profit [EUR]")
    plt.title(f"BESS Daily Profit - Last 5 Days (Total: EUR {total:,.2f})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ── Tab 3: Price Analysis ────────────────────────────────────────────────────

with tab3:
    st.header("Price Analysis")
    df = _prices()

    st.subheader("Price Duration Curve")
    sorted_prices = df["price_eur_mwh"].sort_values(ascending=False).values
    pct_hours = (np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)) * 100
    avg_all = df["price_eur_mwh"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pct_hours, sorted_prices)
    ax.fill_between(pct_hours, sorted_prices, 0, alpha=0.2)
    ax.axhline(avg_all, color="orange", linestyle="--", label=f"Average: {avg_all:.2f}")
    ax.set_xlabel("Percentage of hours"); ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title("Price Duration Curve"); ax.legend(); ax.grid(True)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Yesterday vs Day Before")
    max_date = df["datetime"].dt.date.max()
    yest_p = df[df["datetime"].dt.date == max_date - pd.Timedelta(days=1)]
    db_p   = df[df["datetime"].dt.date == max_date - pd.Timedelta(days=2)]
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(yest_p["datetime"], yest_p["price_eur_mwh"], label="Yesterday")
    ax2.plot(db_p["datetime"],   db_p["price_eur_mwh"],   label="Day Before")
    ax2.axhline(yest_p["price_eur_mwh"].max(), color="r", linestyle="--", alpha=0.6, label="Yesterday Max/Min")
    ax2.axhline(yest_p["price_eur_mwh"].min(), color="r", linestyle="--", alpha=0.6)
    ax2.axhline(db_p["price_eur_mwh"].max(),   color="g", linestyle="--", alpha=0.6, label="Day Before Max/Min")
    ax2.axhline(db_p["price_eur_mwh"].min(),   color="g", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Time"); ax2.set_ylabel("Price [EUR/MWh]")
    ax2.set_title("Yesterday vs Day Before"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.subheader("Negative Prices - Last 7 Days")
    start_7d = df["datetime"].max() - pd.Timedelta(days=7)
    last7_neg = df[df["datetime"] >= start_7d].copy()
    neg_only = last7_neg["price_eur_mwh"].where(last7_neg["price_eur_mwh"] < 0)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(last7_neg["datetime"], last7_neg["price_eur_mwh"], color="grey", alpha=0.35, label="Price")
    ax3.plot(last7_neg["datetime"], neg_only, color="darkblue", label="Negative Prices")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.set_xlabel("Time"); ax3.set_ylabel("Price [EUR/MWh]")
    ax3.set_title("Negative Prices: Last 7 Days"); ax3.legend(); ax3.grid(True)
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    neg_df = last7_neg[last7_neg["price_eur_mwh"] < 0][["datetime", "price_eur_mwh"]]
    if not neg_df.empty:
        st.dataframe(neg_df, use_container_width=True)
        buf_neg = BytesIO()
        neg_df.to_excel(buf_neg, index=False, engine="xlsxwriter")
        st.download_button("Download Negative Prices", buf_neg.getvalue(), "negative_prices.xlsx")

    st.subheader("Monthly Peak / Off-Peak Averages")
    df_po = df.copy()
    df_po["year_month"] = df_po["datetime"].dt.to_period("M")
    df_po["peak"]    = df_po["price_eur_mwh"].where(df_po["datetime"].dt.hour.between(8, 20))
    df_po["offpeak"] = df_po["price_eur_mwh"].where(~df_po["datetime"].dt.hour.between(8, 20))
    monthly_avg = df_po.groupby("year_month").agg({"peak": "mean", "offpeak": "mean"}).reset_index()
    monthly_avg["mid_date"] = monthly_avg["year_month"].apply(
        lambda p: p.to_timestamp() + pd.Timedelta(days=14)
    )
    offset = pd.Timedelta(days=6)
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.plot(df_po["datetime"], df_po["price_eur_mwh"], color="grey", alpha=0.25, linewidth=0.6, label="Price")
    ax4.bar(monthly_avg["mid_date"] - offset, monthly_avg["peak"],    width=11, color="red",  alpha=0.7, label="Avg Peak (8-20)")
    ax4.bar(monthly_avg["mid_date"] + offset, monthly_avg["offpeak"], width=11, color="blue", alpha=0.7, label="Avg Off-peak")
    ax4.set_xlabel("Time"); ax4.set_ylabel("Price [EUR/MWh]")
    ax4.set_title("Monthly Peak / Off-Peak Averages"); ax4.legend()
    ax4.grid(axis="y", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    buf_po = BytesIO()
    monthly_avg[["year_month", "peak", "offpeak"]].astype({"year_month": str}).to_excel(
        buf_po, index=False, engine="xlsxwriter"
    )
    st.download_button("Download Peak/Off-Peak Data", buf_po.getvalue(), "peak_offpeak.xlsx")


# ── Tab 4: Weekly Report ─────────────────────────────────────────────────────

with tab4:
    st.header("Weekly Report")

    with st.spinner("Computing weekly BESS dispatch..."):
        (last_mon, last_fri, prev_mon, prev_fri,
         last_label, prev_label, last_daily, prev_daily, both_wks,
         lw_c, lw_d, pw_c, pw_d) = _weekly_computation()

    def _summary_row(label, period, daily_df, week_df):
        best  = daily_df.loc[daily_df["BESS Optimal P&L [EUR]"].idxmax()]
        worst = daily_df.loc[daily_df["BESS Optimal P&L [EUR]"].idxmin()]
        return {
            "Week": label, "Period": period,
            "Avg DA Price [EUR/MWh]":    round(week_df["price_eur_mwh"].mean(), 2),
            "Total BESS P&L [EUR]":      round(daily_df["BESS Optimal P&L [EUR]"].sum(), 2),
            "Best Day":                  str(best["Date"]),
            "Best Day P&L [EUR]":        round(best["BESS Optimal P&L [EUR]"], 2),
            "Worst Day":                 str(worst["Date"]),
            "Worst Day P&L [EUR]":       round(worst["BESS Optimal P&L [EUR]"], 2),
            "Avg Daily Spread [EUR/MWh]": round(daily_df["Spread [EUR/MWh]"].mean(), 2),
            "Total Negative Hours":      int(daily_df["Negative Price Hours"].sum()),
            "Avg Wind [MWh]":            round(week_df["wind_generation"].mean(), 2),
            "Avg Solar [MWh]":           round(week_df["solar_generation"].mean(), 2),
        }

    prev_wk_df = both_wks[both_wks["datetime"].dt.date < last_mon.date()]
    last_wk_df = both_wks[both_wks["datetime"].dt.date >= last_mon.date()]

    summary = pd.DataFrame([
        _summary_row(prev_label, f"{prev_mon.date()} - {prev_fri.date()}", prev_daily, prev_wk_df),
        _summary_row(last_label, f"{last_mon.date()} - {last_fri.date()}", last_daily, last_wk_df),
    ])

    st.subheader("Weekly Summary")
    st.dataframe(summary.T.astype(str), use_container_width=True)

    daily_detail = pd.concat([prev_daily, last_daily], ignore_index=True)
    st.subheader("Daily Detail")
    st.dataframe(daily_detail, use_container_width=True)

    buf_wk = BytesIO()
    with pd.ExcelWriter(buf_wk, engine="xlsxwriter") as writer:
        daily_detail.to_excel(writer, sheet_name="Daily Detail",   index=False)
        summary.to_excel(writer,      sheet_name="Weekly Summary", index=False)
    st.download_button("Download Weekly Report", buf_wk.getvalue(), "weekly_summary.xlsx")

    def _unzip(pts):
        return zip(*pts) if pts else ([], [])

    all_dates = sorted(both_wks["datetime"].dt.date.unique())
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for i, date in enumerate(all_dates):
        color = "#f0f0f0" if i % 2 == 0 else "#ddeeff"
        ax1.axvspan(pd.Timestamp(date), pd.Timestamp(date) + pd.Timedelta(days=1),
                    alpha=0.5, color=color, zorder=0)

    last_mid = pd.to_datetime(last_daily["Date"]) + pd.Timedelta(hours=12)
    prev_mid = pd.to_datetime(prev_daily["Date"]) + pd.Timedelta(hours=12)
    ax1.plot(last_mid, last_daily["BESS Optimal P&L [EUR]"], marker="o", color="steelblue",  label=last_label)
    ax1.plot(prev_mid, prev_daily["BESS Optimal P&L [EUR]"], marker="o", color="darkorange", label=prev_label)
    ax2.plot(both_wks["datetime"], both_wks["price_eur_mwh"],
             color="dimgray", alpha=0.35, linewidth=0.6, label="Price [EUR/MWh]")

    t, p = _unzip(lw_c); ax2.scatter(list(t), list(p), color="royalblue",  zorder=5, s=35, marker="^", label="Charge - Last Wk")
    t, p = _unzip(lw_d); ax2.scatter(list(t), list(p), color="crimson",    zorder=5, s=35, marker="v", label="Discharge - Last Wk")
    t, p = _unzip(pw_c); ax2.scatter(list(t), list(p), color="limegreen",  zorder=5, s=35, marker="^", label="Charge - Prev Wk")
    t, p = _unzip(pw_d); ax2.scatter(list(t), list(p), color="darkorange", zorder=5, s=35, marker="v", label="Discharge - Prev Wk")

    tick_pos = [pd.Timestamp(d) + pd.Timedelta(hours=12) for d in all_dates]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels([pd.Timestamp(d).strftime("%a\n%d %b") for d in all_dates], fontsize=8)
    ax1.set_xlim(
        pd.Timestamp(prev_mon.date()) - pd.Timedelta(hours=12),
        pd.Timestamp(last_fri.date()) + pd.Timedelta(hours=36),
    )
    ax1.set_xlabel("Date"); ax1.set_ylabel("BESS Optimal P&L [EUR]"); ax2.set_ylabel("Price [EUR/MWh]")
    plt.title("Daily BESS Optimal P&L: Last Week vs Previous Week")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ── Tab 5: Wind Forecast ─────────────────────────────────────────────────────

with tab5:
    st.header("Wind Forecast vs Actual")
    wf = _wind_forecast_df()

    wf["error_mwh"]     = wf["actual_mwh"] - wf["forecast_mwh"]
    wf["abs_error_mwh"] = wf["error_mwh"].abs()

    daily_wf = (
        wf.groupby([wf["datetime"].dt.date, "source"])
        .agg(Mean_Error=("error_mwh", "mean"), MAE=("abs_error_mwh", "mean"))
        .round(2).reset_index()
        .rename(columns={"datetime": "date"})
    )

    sources = wf["source"].unique()

    fig, axes = plt.subplots(len(sources), 1, figsize=(14, 4 * len(sources)), sharex=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, (source, group) in zip(axes, wf.groupby("source")):
        ax.plot(group["datetime"], group["actual_mwh"],   label="Actual",   linewidth=0.8)
        ax.plot(group["datetime"], group["forecast_mwh"], label="Forecast", linewidth=0.8, linestyle="--")
        ax.set_title(source); ax.set_ylabel("MWh")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)
    axes[-1].set_xlabel("Datetime")
    plt.suptitle("Wind Forecast vs Actual", y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig2, axes2 = plt.subplots(len(sources), 1, figsize=(14, 4 * len(sources)), sharex=True)
    if len(sources) == 1:
        axes2 = [axes2]
    for ax, (source, group) in zip(axes2, daily_wf.groupby("source")):
        ax.bar(group["date"], group["Mean_Error"],
               color=["seagreen" if v >= 0 else "tomato" for v in group["Mean_Error"]],
               label="Mean Error [MWh]")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(source + " - Daily Mean Error"); ax.set_ylabel("MWh")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)
    axes2[-1].set_xlabel("Date")
    plt.suptitle("Daily Mean Forecast Error (Actual - Forecast)", y=1.01)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.subheader("Accuracy Summary")
    st.dataframe(
        daily_wf.groupby("source")[["Mean_Error", "MAE"]].mean().round(2),
        use_container_width=True
    )

    # --- Plot correlation plot between prices_eur_mwh and solar/wind forecast error ---
    # for each week on a lowest granularity(15 min) ---
    def prices_eur_mwh_forecast_solar_wind_correlation(df, solar_forecast_df, wind_forecast_df):
        plt.figure(figsize(14,6))
        prices = df["prices_eur_mwh"]
        solar_error = solar_forecast_df["error_mwh"]
        wind_error = wind_forecast_df["error_mwh"]
        
        plt.scatter(prices, solar_error, alpha=0.5, label="Solar Error")
        plt.scatter(prices, wind_error, alpha=0.5, label="Wind Error")
        plt.xlabel("Prices [EUR/MWh]")
        plt.ylabel("Forecast Error [MWh]")
        plt.title("Correlation between Prices and Forecast Error")
        corr_solar = df["prices_eur_mwh"].corr(solar_forecast_df["error_mwh"])
        corr_wind = df["prices_eur_mwh"].corr(wind_forecast_df["error_mwh"])
        plt.text(0.5, 0.9, f"Solar Correlation: {corr_solar:.2f}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.85, f"Wind Correlation: {corr_wind:.2f}", transform=plt.gca().transAxes)
        plt.legend()
        plt.grid(True)
        plt.show()  

        df_result = pd.DataFrame({
            "Prices [EUR/MWh]": prices,
            "Solar Error [MWh]": solar_error,
            "Wind Error [MWh]": wind_error
        })
        
    buf_wf = BytesIO()
    with pd.ExcelWriter(buf_wf, engine="xlsxwriter") as writer:
        wf.to_excel(writer,       sheet_name="Data",    index=False)
        daily_wf.to_excel(writer, sheet_name="Summary", index=False)
    st.download_button("Download Wind Forecast Data", buf_wf.getvalue(), "wind_forecast_accuracy.xlsx")
