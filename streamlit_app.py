"""
Fitness Tracker Analytics — Streamlit app
=======================================================
Built directly on the literally-merged dataset files (daily_merged.csv,
hourly_merged.csv — outer-joined, no values corrected or altered) and on
the chart set from the project reference deck (weekday patterns,
per-participant heatmaps, activity-category breakdown, calories-by-hour).

The app contains:
  - An interactive Power BI-style dashboard (Plotly)
  - A built-in SQL Lab (SQLite in-memory, pre-written + custom queries)
  - Per-participant heatmaps (steps / sedentary minutes / calories)
  - A sleep & weight snapshot
  - Customer-facing recommendations
  - A data & methodology tab documenting the merge approach

RUN LOCALLY (keep daily_merged.csv and hourly_merged.csv in the same folder):
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Data: Fitbit fitness-tracker export, 33 users, 12 Apr – 12 May 2016
(public CC0 dataset, distributed via Amazon Mechanical Turk).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Fitness Tracker Analytics", page_icon="\U0001F3C3", layout="wide")

# ============================================================== data files
# The two merged CSVs sit next to this script (same folder in the GitHub repo).
DATA_DIR = Path(__file__).resolve().parent
DAILY_CSV = DATA_DIR / "daily_merged.csv"
HOURLY_CSV = DATA_DIR / "hourly_merged.csv"


WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NAVY, TEAL, AMBER, RED, PURPLE, GREY = "#2E4057", "#29A0B1", "#F3A712", "#E4572E", "#5B5F97", "#9AA5B1"
PALETTE = [NAVY, TEAL, AMBER, RED, PURPLE, "#A8C686", "#7A9E9F", "#C9D6EA"]

# Card surface + text colors are set explicitly (not left to the Streamlit theme) so they
# render correctly whether the viewer is in light or dark mode — a card that inherits the
# theme's background but hardcodes its own text color (or vice versa) goes unreadable.
CARD_BG, CARD_BORDER, TXT_MAIN, TXT_SUB = "#141C2B", "#26344A", "#F3F6FA", "#93A0B4"

st.markdown(f"""<style>
div[data-testid="stMetric"] {{ display:none; }}
.bi-banner{{background:linear-gradient(90deg,{NAVY},{TEAL});color:#fff;padding:16px 22px;border-radius:10px;margin-bottom:14px}}
.bi-banner h2{{margin:0;color:#fff}} .bi-banner span{{opacity:.85;font-size:.85rem}}

.kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
.kpi-card{{flex:1;min-width:150px;background:{CARD_BG};border:1px solid {CARD_BORDER};border-left:4px solid {TEAL};
  border-radius:10px;padding:12px 16px}}
.kpi-label{{color:{TXT_SUB};font-size:0.78rem;font-weight:500;margin-bottom:4px;text-transform:uppercase;letter-spacing:.02em}}
.kpi-value{{color:{TXT_MAIN};font-size:1.55rem;font-weight:700;line-height:1.2}}
.kpi-sub{{color:{TXT_SUB};font-size:0.75rem;margin-top:2px}}

.section-card{{background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:10px;padding:16px 18px 6px 18px;margin-bottom:16px}}
.section-title{{color:{TXT_MAIN};font-size:1rem;font-weight:700;margin-bottom:2px}}
.section-caption{{color:{TXT_SUB};font-size:0.8rem;margin-bottom:10px}}

.rec{{background:{CARD_BG};border:1px solid {CARD_BORDER};border-left:5px solid {TEAL};border-radius:10px;
  padding:14px 18px;margin-bottom:12px}}
.rec h4{{margin:0 0 6px 0;color:{TXT_MAIN}}} .rec p{{margin:0;color:{TXT_SUB};font-size:0.95rem}}
.rec .tag{{display:inline-block;background:rgba(41,160,177,0.18);color:{TEAL};font-size:0.72rem;font-weight:600;
  padding:2px 8px;border-radius:999px;margin-bottom:6px}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
# Color stops for the heatmaps (same palettes as ColorBrewer). They are defined here, instead
# of using plotly's built-in names, so we can also work out how dark each cell is and pick
# a readable text color for the number printed on it.
HEAT_SCALES = {
    "YlOrBr": ["#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
    "Reds": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
    "Blues": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
    "RdBu": ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
             "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"],
}


def annotated_heatmap(table_df, colorscale, value_fmt=".0f", title="", height=680,
                      colorbar_title="", zmin=None, zmax=None, label_prefix=""):
    """
    Heatmap that prints the real value inside every cell (not just a color).
    plotly picks a readable text color for each cell on its own, and cells with no
    data (NaN) are simply left empty instead of showing a misleading 0.

    label_prefix is put in front of the row labels (for example "ID ") so plotly always
    treats them as text and never as numbers.
    """
    stops = HEAT_SCALES[colorscale]
    z = table_df.values.astype(float)
    x_labels = [str(c) for c in table_df.columns]
    y_labels = [f"{label_prefix}{i}" for i in table_df.index]

    # plotly draws the first row at the bottom, so flip the rows to keep the first row on top
    z = z[::-1]
    y_labels = y_labels[::-1]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        zmin=zmin, zmax=zmax,
        colorscale=[[k / (len(stops) - 1), c] for k, c in enumerate(stops)],
        colorbar=dict(title=colorbar_title),
        xgap=2, ygap=2,
        texttemplate="%{z:" + value_fmt + "}",
        textfont=dict(size=12),
        hoverongaps=False,
    ))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=48, b=10))
    fig.update_xaxes(side="top")
    return fig


def kpi_row(items):
    """items: list of (label, value, sub) tuples. sub is optional, pass '' to omit."""
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        + (f'<div class="kpi-sub">{sub}</div>' if sub else "") + "</div>"
        for label, value, sub in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def section(title, caption=""):
    st.markdown(f'<div class="section-title">{title}</div>'
                + (f'<div class="section-caption">{caption}</div>' if caption else ""),
                unsafe_allow_html=True)


DARK_PLOT_BG = "#0E1520"
DARK_PAPER_BG = "#141C2B"
DARK_GRID = "#243043"
DARK_FONT = "#D7DEE8"


def show(fig, h=360, key=None):
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=48, b=10), colorway=PALETTE,
        plot_bgcolor=DARK_PLOT_BG, paper_bgcolor=DARK_PAPER_BG,
        font=dict(color=DARK_FONT, size=12),
        title_font=dict(color=TXT_MAIN, size=14),
        legend=dict(font=dict(color=DARK_FONT)),
        xaxis=dict(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, color=DARK_FONT),
        yaxis=dict(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, color=DARK_FONT),
    )
    fig.update_xaxes(gridcolor=DARK_GRID, color=DARK_FONT)
    fig.update_yaxes(gridcolor=DARK_GRID, color=DARK_FONT)
    try:
        st.plotly_chart(fig, width="stretch", key=key, theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)


def trend_line(fig, x, y, color="#F3F6FA"):
    ok = x.notna() & y.notna()
    if ok.sum() > 2:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                  line=dict(color=color, dash="dash")))
    return fig


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


@st.cache_data(show_spinner="Loading data...")
def load():
    """Read the merged CSVs and add analysis-only derived columns.
    Nothing here touches the original source values (steps, calories, sedentary
    minutes, etc. are exactly what the merge step produced) — every column added
    below is a computed convenience for grouping/plotting, matching the same
    "don't correct the data" rule the merge step followed.
    Cached with st.cache_data so this read/derive work only runs once per
    session, not on every widget interaction.
    """
    d = pd.read_csv(DAILY_CSV)
    h = pd.read_csv(HOURLY_CSV)

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Weekday"] = pd.Categorical(d["Date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    d["IsWeekend"] = d["Weekday"].isin(["Saturday", "Sunday"])          # used by the weekday-vs-weekend chart
    d["TotalActiveMinutes"] = d["LightlyActiveMinutes"] + d["FairlyActiveMinutes"] + d["VeryActiveMinutes"]
    d["SedentaryHours"] = d["SedentaryMinutes"] / 60.0
    d["SleepHours"] = d["TotalMinutesAsleep"] / 60.0

    h["ActivityHour"] = pd.to_datetime(h["ActivityHour"], errors="coerce")
    h["Date"] = h["ActivityHour"].dt.date
    h["Hour"] = h["ActivityHour"].dt.hour
    h["Weekday"] = pd.Categorical(h["ActivityHour"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    return d, h


@st.cache_resource
def sql_db():
    d, h = load()
    con = sqlite3.connect(":memory:", check_same_thread=False)
    dd, hh = d.copy(), h.copy()
    dd["Date"] = dd["Date"].dt.strftime("%Y-%m-%d")
    hh["ActivityHour"] = hh["ActivityHour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    hh["Date"] = hh["Date"].astype(str)
    for c in dd.select_dtypes("bool"):
        dd[c] = dd[c].astype(int)
    if "IsManualReport" in dd.columns:
        dd["IsManualReport"] = dd["IsManualReport"].map({True: 1, False: 0, "True": 1, "False": 0})
    dd.to_sql("daily", con, index=False)
    hh.to_sql("hourly", con, index=False)
    con.execute("PRAGMA query_only = ON")
    return con


# SQLite has no native "order weekdays Mon->Sun" concept — this CASE expression maps
# each weekday name to its calendar position so ORDER BY sorts correctly instead of
# alphabetically (used by query #2 below).
WD_CASE = ("CASE Weekday WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3 "
           "WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6 ELSE 7 END")

# Each tuple is (label shown in the dropdown, one-line caption, the SQL itself).
# Deliberately spans a range of SQL techniques (aggregation, window functions,
# CTEs, self-joins) so the SQL Lab doubles as a small reference, not just a demo.
QUERIES = [
    ("1. Dataset overview", "Size and coverage of the merged daily table (no rows dropped by the merge).",
     """SELECT COUNT(*) AS participant_days, COUNT(DISTINCT Id) AS participants,
       MIN(Date) AS first_day, MAX(Date) AS last_day,
       ROUND(AVG(TotalSteps), 0) AS avg_steps, ROUND(AVG(Calories), 0) AS avg_calories,
       COUNT(TotalMinutesAsleep) AS days_with_sleep_log, COUNT(WeightKg) AS days_with_weight_log
FROM daily;"""),
    ("2. Average steps by weekday", "The core weekday chart from the dashboard, as SQL.",
     f"""SELECT Weekday, COUNT(*) AS days, ROUND(AVG(TotalSteps), 0) AS avg_steps,
       ROUND(AVG(Calories), 0) AS avg_calories, ROUND(AVG(SedentaryHours), 2) AS avg_sedentary_hours
FROM daily GROUP BY Weekday ORDER BY {WD_CASE};"""),
    ("3. Top 10 most active participants", "Ranked by average daily steps.",
     """SELECT RANK() OVER (ORDER BY AVG(TotalSteps) DESC) AS rnk, Id,
       COUNT(*) AS days_logged, ROUND(AVG(TotalSteps), 0) AS avg_steps,
       ROUND(AVG(TotalActiveMinutes), 1) AS avg_active_min
FROM daily GROUP BY Id ORDER BY rnk LIMIT 10;"""),
    ("4. Sedentary-risk participants", "Participants averaging 16+ sedentary hours/day.",
     """SELECT Id, COUNT(*) AS days, ROUND(AVG(SedentaryHours), 1) AS avg_sedentary_h,
       ROUND(AVG(TotalSteps), 0) AS avg_steps
FROM daily GROUP BY Id HAVING AVG(SedentaryHours) >= 16 ORDER BY avg_sedentary_h DESC;"""),
    ("5. Hourly activity profile", "Average steps, calories and intensity for every hour of the day.",
     """SELECT Hour, ROUND(AVG(StepTotal), 1) AS avg_steps, ROUND(AVG(Calories), 1) AS avg_calories,
       ROUND(AVG(TotalIntensity), 2) AS avg_intensity
FROM hourly GROUP BY Hour ORDER BY Hour;"""),
    ("6. Sleep duration vs activity", "Are participants who sleep more also more active? Nights bucketed by hours asleep.",
     """SELECT CASE WHEN TotalMinutesAsleep < 360 THEN '1) < 6 h' WHEN TotalMinutesAsleep < 420 THEN '2) 6-7 h'
             WHEN TotalMinutesAsleep < 480 THEN '3) 7-8 h' ELSE '4) 8 h+' END AS sleep_bucket,
       COUNT(*) AS nights, ROUND(AVG(TotalSteps), 0) AS avg_steps,
       ROUND(AVG(TotalActiveMinutes), 1) AS avg_active_min
FROM daily WHERE TotalMinutesAsleep IS NOT NULL GROUP BY sleep_bucket ORDER BY sleep_bucket;"""),
    ("7. Time lost awake in bed", "Participants who spend the most time in bed awake (TimeInBed - MinutesAsleep).",
     """SELECT Id, COUNT(*) AS nights, ROUND(AVG(TotalMinutesAsleep) / 60.0, 2) AS avg_sleep_h,
       ROUND(AVG(TotalTimeInBed - TotalMinutesAsleep), 1) AS avg_min_awake_in_bed
FROM daily WHERE TotalMinutesAsleep IS NOT NULL GROUP BY Id HAVING nights >= 5
ORDER BY avg_min_awake_in_bed DESC LIMIT 10;"""),
    ("8. Weight change per participant", "First vs last recorded weight (ROW_NUMBER window) — logging is sparse.",
     """WITH w AS (SELECT Id, Date, WeightKg, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY Date ASC) AS rn_first,
           ROW_NUMBER() OVER (PARTITION BY Id ORDER BY Date DESC) AS rn_last
    FROM daily WHERE WeightKg IS NOT NULL)
SELECT f.Id, ROUND(f.WeightKg, 1) AS first_kg, ROUND(l.WeightKg, 1) AS last_kg,
       ROUND(l.WeightKg - f.WeightKg, 1) AS change_kg
FROM w f JOIN w l ON f.Id = l.Id AND f.rn_first = 1 AND l.rn_last = 1 ORDER BY change_kg;"""),
    ("9. 7-day moving average of steps", "Community trend smoothed with a window frame.",
     """WITH d AS (SELECT Date, AVG(TotalSteps) AS avg_steps FROM daily GROUP BY Date)
SELECT Date, ROUND(avg_steps, 0) AS avg_steps,
       ROUND(AVG(avg_steps) OVER (ORDER BY Date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0) AS moving_avg_7d
FROM d ORDER BY Date;"""),
    ("10. Consistency of each participant", "Coefficient of variation of daily steps (lower = steadier routine).",
     """SELECT Id, COUNT(*) AS days, ROUND(AVG(TotalSteps), 0) AS avg_steps,
       ROUND(SQRT(AVG(TotalSteps*TotalSteps) - AVG(TotalSteps)*AVG(TotalSteps))
             / NULLIF(AVG(TotalSteps),0), 2) AS coeff_of_variation
FROM daily GROUP BY Id HAVING days >= 14 ORDER BY coeff_of_variation;"""),
    ("11. Calories by vigorous-activity bucket", "How much extra energy does vigorous activity burn?",
     """SELECT CASE WHEN VeryActiveMinutes = 0 THEN '0 min' WHEN VeryActiveMinutes < 30 THEN '1-29 min'
             WHEN VeryActiveMinutes < 60 THEN '30-59 min' ELSE '60+ min' END AS very_active_bucket,
       COUNT(*) AS days, ROUND(AVG(Calories), 0) AS avg_calories, ROUND(AVG(TotalSteps), 0) AS avg_steps
FROM daily GROUP BY very_active_bucket ORDER BY MIN(VeryActiveMinutes);"""),
    ("12. Best day of each participant", "Rank each participant's days by steps and keep the best one.",
     """SELECT Id, Date, Weekday, TotalSteps, Calories
FROM (SELECT *, RANK() OVER (PARTITION BY Id ORDER BY TotalSteps DESC) AS r FROM daily) WHERE r = 1
ORDER BY TotalSteps DESC LIMIT 15;"""),
]

# ------------------------------------------------------------------ load + sidebar filters
# daily_all / hourly_all: the full, unfiltered data (used as-is by the SQL Lab).
# d / h below are the sidebar-filtered views every other tab actually plots.
daily_all, hourly_all = load()

st.sidebar.title("\U0001F3C3 Fitness Tracker Analytics")
st.sidebar.caption("Dashboard, participant heatmaps, SQL lab and recommendations — built on the merged, "
                    "uncorrected Fitbit export.")
dmin, dmax = daily_all.Date.min().date(), daily_all.Date.max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
d0, d1 = (pd.Timestamp(dr[0]), pd.Timestamp(dr[1])) if isinstance(dr, tuple) and len(dr) == 2 \
    else (pd.Timestamp(dmin), pd.Timestamp(dmax))

all_ids = sorted(daily_all.Id.unique())
ids_sel = st.sidebar.multiselect("Participants (Id)", all_ids, default=all_ids)
weekdays_sel = st.sidebar.multiselect("Weekdays", WEEKDAY_ORDER, default=WEEKDAY_ORDER)

d = daily_all[daily_all.Date.between(d0, d1) & daily_all.Id.isin(ids_sel) & daily_all.Weekday.isin(weekdays_sel)]
h = hourly_all[(hourly_all.ActivityHour >= d0) & (hourly_all.ActivityHour <= d1 + pd.Timedelta(days=1))
               & hourly_all.Id.isin(ids_sel) & hourly_all.Weekday.isin(weekdays_sel)]

if d.empty:
    st.warning("No data for the current filters.")
    st.stop()

st.sidebar.markdown(
    f'<div class="kpi-card" style="min-width:0"><div class="kpi-label">Rows in view</div>'
    f'<div class="kpi-value" style="font-size:1.15rem">{len(d):,} participant-days</div></div>',
    unsafe_allow_html=True)

st.title("Fitness Tracker Analytics")
st.caption(f"{d.Id.nunique()} participants \u00b7 {d.Date.min():%d %b %Y} \u2013 {d.Date.max():%d %b %Y} \u00b7 "
           "merged from the project's 19 raw tracker export files (day \u2192 hour grain) \u2014 values unaltered, "
           "only outer-joined")

tabs = st.tabs(["\U0001F4C8 BI Dashboard", "\U0001F465 Participant Heatmaps", "\U0001F634 Sleep & Weight",
                "\U0001F9EE SQL Analysis", "\u2705 Recommendations", "\U0001F5C2 Data & Methodology"])

# ================================================================ 1. BI Dashboard
with tabs[0]:
    st.markdown(f"""<div class="bi-banner"><h2>Overview</h2>
    <span>Key numbers for the current filter selection</span></div>""", unsafe_allow_html=True)

    kpi_row([
        ("Avg. daily steps", f"{d.TotalSteps.mean():,.0f}", ""),
        ("Avg. daily calories", f"{d.Calories.mean():,.0f}", ""),
        ("Avg. active minutes", f"{d.TotalActiveMinutes.mean():,.0f} min", ""),
        ("Avg. sedentary time", f"{d.SedentaryHours.mean():,.1f} h", ""),
        ("Avg. sleep", f"{d.SleepHours.mean():,.1f} h" if d.SleepHours.notna().any() else "n/a",
         "" if d.SleepHours.notna().any() else "no log in range"),
        ("Participants", f"{d.Id.nunique()}", f"of {daily_all.Id.nunique()} total"),
    ])

    section("Weekly activity pattern", "How steps, active time and calories move through the week")
    left, right = st.columns(2)
    with left:
        # Group by weekday and average across every participant/date in the current
        # filter, then reindex to force calendar order (Mon->Sun) — groupby on its own
        # would sort alphabetically (Friday, Monday, Saturday, ...)
        steps_wd = d.groupby("Weekday", observed=True)["TotalSteps"].mean().reindex(WEEKDAY_ORDER)
        fig = go.Figure(go.Bar(x=steps_wd.index, y=steps_wd.values, marker_color=TEAL,
                                text=steps_wd.round(0), textposition="outside"))
        fig.update_layout(title="Average Steps Taken Through Weekdays", yaxis_title="Avg. Total Steps")
        show(fig)
    with right:
        # Average minutes spent in each Fitbit activity-intensity bucket, across the
        # whole filtered selection — greys out "Sedentary" so the three genuinely
        # active categories are the visual focus
        cat_means = d[["LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes", "SedentaryMinutes"]].mean()
        cat_means.index = ["Lightly Active", "Fairly Active", "Very Active", "Sedentary"]
        colors = [AMBER, AMBER, AMBER, GREY]
        fig = go.Figure(go.Bar(x=cat_means.index, y=cat_means.values, marker_color=colors,
                                text=cat_means.round(1), textposition="outside"))
        fig.update_layout(title="Time Spent in Different Activity Categories", yaxis_title="Avg. Minutes")
        show(fig)

    left2, right2 = st.columns(2)
    with left2:
        # Two very different scales (minutes vs. calories) in one chart, so calories
        # gets its own secondary y-axis ("yaxis2") rather than being squashed flat
        # against the active-minutes bars
        combo = d.groupby("Weekday", observed=True)[["Calories", "TotalActiveMinutes"]].mean().reindex(WEEKDAY_ORDER)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=combo.index, y=combo["TotalActiveMinutes"], name="Avg Total Active Minutes",
                              marker_color=NAVY, yaxis="y"))
        fig.add_trace(go.Scatter(x=combo.index, y=combo["Calories"], name="Avg Calories",
                                  mode="lines+markers", marker_color=AMBER, yaxis="y2"))
        fig.update_layout(title="Avg. Calories Burnt During Active Time",
                           yaxis=dict(title="Avg. Total Active Minutes"),
                           yaxis2=dict(title="Avg. Calories", overlaying="y", side="right"),
                           legend=dict(orientation="h", y=1.15))
        show(fig)
    with right2:
        # Hourly data comes from the separate hourly table (h, not d) — grouped by
        # the Hour-of-day column and reindexed 0-23 so every hour shows even if a
        # given hour happens to have no rows in the current filter
        cal_hr = h.groupby("Hour")["Calories"].mean().reindex(range(24))
        fig = go.Figure(go.Bar(x=cal_hr.index, y=cal_hr.values, marker_color=TEAL))
        fig.update_layout(title="Calories Burnt Each Hour", xaxis_title="Activity Hour", yaxis_title="Avg. Calories")
        fig.update_xaxes(dtick=1)
        show(fig)

    section("Trend over time", "Day-by-day steps for the selected range, smoothed with a 7-day moving average")
    # Raw daily averages are noisy; a 7-day rolling mean smooths that out so a real
    # trend (if any) is easier to see than in the thin grey daily line underneath it
    daily_trend = d.groupby("Date")["TotalSteps"].mean().sort_index()
    moving_avg = daily_trend.rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_trend.index, y=daily_trend.values, mode="lines", name="Daily avg",
                              line=dict(color=GREY, width=1)))
    fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode="lines", name="7-day moving avg",
                              line=dict(color=TEAL, width=3)))
    fig.update_layout(title="Average Steps Over Time", yaxis_title="Avg. Total Steps",
                       legend=dict(orientation="h", y=1.15))
    show(fig, h=340)

    section("Distribution & relationships", "How steps, calories and participants relate to one another")
    left3, right3 = st.columns(2)
    with left3:
        # Distribution (not just the average) of daily steps, with a dashed line
        # marking the mean so skew is visible at a glance
        fig = px.histogram(d, x="TotalSteps", nbins=30, color_discrete_sequence=[TEAL])
        fig.add_vline(x=d.TotalSteps.mean(), line_dash="dash", line_color=AMBER,
                      annotation_text="mean", annotation_font_color=DARK_FONT)
        fig.update_layout(title="Distribution of Daily Steps", xaxis_title="Total Steps", yaxis_title="Participant-days")
        show(fig, h=340)
    with right3:
        # Scatter + fitted trend line (trend_line() does a simple linear regression
        # via numpy.polyfit) to check whether active minutes actually predict calories
        fig = px.scatter(d, x="TotalActiveMinutes", y="Calories", color_discrete_sequence=[TEAL], opacity=0.6)
        fig = trend_line(fig, d["TotalActiveMinutes"], d["Calories"])
        fig.update_layout(title="Active Minutes vs. Calories Burnt", xaxis_title="Total Active Minutes",
                           yaxis_title="Calories", legend=dict(orientation="h", y=1.15))
        show(fig, h=340)

    left4, right4 = st.columns(2)
    with left4:
        # Ranks participants by their own average steps (not by a single best day),
        # so this reflects a sustained pattern rather than one outlier day
        top10 = d.groupby("Id", observed=True)["TotalSteps"].mean().sort_values(ascending=False).head(10)
        fig = go.Figure(go.Bar(x=top10.values[::-1], y=top10.index.astype(str)[::-1], orientation="h",
                                marker_color=TEAL, text=top10.values[::-1].round(0), textposition="outside"))
        fig.update_layout(title="Top 10 Participants by Avg. Steps", xaxis_title="Avg. Total Steps")
        show(fig, h=380)
    with right4:
        corr_cols = ["TotalSteps", "Calories", "TotalActiveMinutes", "SedentaryMinutes", "SleepHours"]
        corr = d[corr_cols].corr().round(2)
        corr.columns = corr.index = ["Steps", "Calories", "Active min", "Sedentary min", "Sleep hrs"]
        # annotated_heatmap() prints the real correlation coefficient in every cell,
        # with auto-contrasted text, instead of relying on the color scale alone —
        # this is the "heatmap shows real measures" fix applied everywhere in the app
        fig = annotated_heatmap(corr, colorscale="RdBu", value_fmt=".2f",
                                 title="Correlation Between Key Metrics", height=380, zmin=-1, zmax=1)
        show(fig, h=380)

    section("Daily rhythm & weekday vs. weekend", "How steps move hour-by-hour, and how weekdays compare to weekends")
    left5, right5 = st.columns(2)
    with left5:
        steps_hr = h.groupby("Hour")["StepTotal"].mean().reindex(range(24))
        fig = go.Figure(go.Scatter(x=steps_hr.index, y=steps_hr.values, mode="lines", fill="tozeroy",
                                    line=dict(color=TEAL, width=2)))
        fig.update_layout(title="Average Steps Taken Each Hour", xaxis_title="Activity Hour", yaxis_title="Avg. Steps")
        fig.update_xaxes(dtick=1)
        show(fig, h=340)
    with right5:
        # One small-multiples chart (4 side-by-side bar pairs via make_subplots)
        # rather than 4 separate charts, so weekday-vs-weekend is easy to compare
        # metric-by-metric in a single glance
        wd_we = d.groupby("IsWeekend", observed=True)[["TotalSteps", "TotalActiveMinutes",
                                                         "SedentaryHours", "SleepHours"]].mean()
        wd_we.index = ["Weekday", "Weekend"]
        metrics = [("TotalSteps", "Steps"), ("TotalActiveMinutes", "Active min"),
                   ("SedentaryHours", "Sedentary hrs"), ("SleepHours", "Sleep hrs")]
        fig = make_subplots(rows=1, cols=4, subplot_titles=[m[1] for m in metrics])
        for i, (col, _) in enumerate(metrics, start=1):
            fig.add_trace(go.Bar(x=["Weekday", "Weekend"], y=wd_we[col], marker_color=[NAVY, TEAL],
                                  text=wd_we[col].round(1), textposition="outside", showlegend=False),
                          row=1, col=i)
        fig.update_layout(title="Weekday vs. Weekend Behaviour")
        show(fig, h=340)

    left6, right6 = st.columns(2)
    with left6:
        fig = px.histogram(d, x="Calories", nbins=30, color_discrete_sequence=[AMBER])
        fig.add_vline(x=d.Calories.mean(), line_dash="dash", line_color=TEAL,
                      annotation_text="mean", annotation_font_color=DARK_FONT)
        fig.update_layout(title="Distribution of Daily Calories", xaxis_title="Calories", yaxis_title="Participant-days")
        show(fig, h=340)
    with right6:
        # A boxplot instead of another bar-of-averages on purpose: the weekday bar
        # chart above only shows the mean, this shows the full spread per weekday so
        # we can see whether day-of-week or individual variation dominates
        fig = px.box(d, x="Weekday", y="SedentaryHours", category_orders={"Weekday": WEEKDAY_ORDER},
                     color_discrete_sequence=[RED])
        fig.update_layout(title="Spread of Sedentary Hours by Weekday (not just the average)",
                           xaxis_title="", yaxis_title="Sedentary Hours")
        show(fig, h=340)

    section("Participant consistency", "Who has a steady routine vs. who swings between very active and very inactive days")
    # Coefficient of variation (std / mean) per participant: lower = a steadier
    # day-to-day routine, higher = swings between very active and very inactive days.
    # Only participants with >=5 logged days in the current filter are shown, so the
    # estimate isn't based on just one or two data points.
    grp = d.groupby("Id", observed=True)["TotalSteps"]
    cv = (grp.std() / grp.mean()).dropna()
    cv = cv[grp.count().reindex(cv.index) >= 5].sort_values()
    if len(cv):
        fig = go.Figure(go.Bar(x=cv.values, y=cv.index.astype(str), orientation="h", marker_color=PURPLE))
        fig.update_layout(title="Consistency of Each Participant's Daily Steps (lower = steadier routine)",
                           xaxis_title="Coefficient of Variation (std / mean)", height=max(320, 22 * len(cv)))
        show(fig, h=max(320, 22 * len(cv)))
    else:
        st.info("Not enough logged days per participant in the current filter to compute consistency.")

# ================================================================ 2. Participant heatmaps
with tabs[1]:
    st.caption("Each row is one participant and each column is a day of the week. "
               "This shows whether activity depends more on the person or on the day.")
    metric = st.radio("Metric", ["Steps", "Sedentary minutes", "Calories"], horizontal=True)
    col_map = {"Steps": "TotalSteps", "Sedentary minutes": "SedentaryMinutes", "Calories": "Calories"}
    cmap_map = {"Steps": "YlOrBr", "Sedentary minutes": "Reds", "Calories": "Blues"}
    col = col_map[metric]

    # A one-line explanation for each metric, so anyone can read the chart without help
    how_to_read = {
        "Steps": "Number in each box = average steps walked on that weekday. "
                 "Darker orange = more steps. The most active participants are at the top.",
        "Sedentary minutes": "Number in each box = average minutes spent sitting or not moving on that weekday "
                             "(1,440 minutes = a full day). Darker red = more time inactive.",
        "Calories": "Number in each box = average calories burned on that weekday. "
                    "Darker blue = more calories burned.",
    }
    st.info("How to read this chart: " + how_to_read[metric] + " An empty box means no data for that day.")

    # rows = participant Id, columns = Weekday, cell value = mean of the chosen metric.
    # reindex forces Mon->Sun column order; a participant with no logged day for a
    # given weekday gets a NaN cell, which annotated_heatmap() renders as blank
    # rather than a misleading zero.
    pivot = d.pivot_table(index="Id", columns="Weekday", values=col, aggfunc="mean", observed=True) \
             .reindex(columns=WEEKDAY_ORDER)
    # annotated_heatmap prints the real averaged number in every cell (auto-contrasted
    # per cell) instead of relying on the reader to judge the value from color alone
    # Most active participants at the top, so the pattern is easy to see at a glance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = annotated_heatmap(pivot, colorscale=cmap_map[metric], value_fmt=".0f",
                             title=f"Average {metric} per participant and weekday",
                             height=max(560, 26 * len(pivot)), colorbar_title=f"Avg. {metric}",
                             label_prefix="ID ")
    show(fig, h=max(560, 26 * len(pivot)))

    section("Ranked list", f"The same participants as the chart above, with their overall average {metric.lower()} per day")
    ranked = pivot.mean(axis=1).sort_values(ascending=False).round(1).rename(f"Avg. {metric}")
    table(ranked.reset_index().rename(columns={"index": "Id"}), hide_index=True)

# ================================================================ 3. Sleep & Weight
with tabs[2]:
    sleep_d = d[d.TotalMinutesAsleep.notna()]
    weight_d = d[d.WeightKg.notna()]

    kpi_row([
        ("Avg. sleep / night", f"{sleep_d.SleepHours.mean():.2f} h" if len(sleep_d) else "n/a", ""),
        ("Avg. time awake in bed", f"{(sleep_d.TotalTimeInBed - sleep_d.TotalMinutesAsleep).mean():.0f} min"
         if len(sleep_d) else "n/a", ""),
        ("Nights logged", f"{len(sleep_d):,}", f"of {len(d):,} participant-days"),
        ("Participants with a weight log", f"{weight_d.Id.nunique()} / {d.Id.nunique()}", ""),
    ])

    section("Sleep patterns", "Distribution of nightly sleep, and whether more sleep tracks with more activity")
    left, right = st.columns(2)
    with left:
        if len(sleep_d):
            fig = px.histogram(sleep_d, x="SleepHours", nbins=20, color_discrete_sequence=[TEAL])
            fig.add_vline(x=7, line_dash="dash", line_color=AMBER,
                          annotation_text="7h guideline", annotation_font_color=DARK_FONT)
            fig.update_layout(title="Distribution of Nightly Sleep (hours)", yaxis_title="Nights")
            show(fig)
        else:
            st.info("No sleep records in the current filter selection.")
    with right:
        if len(sleep_d):
            fig = px.scatter(sleep_d, x="SleepHours", y="TotalSteps", color_discrete_sequence=[TEAL], opacity=0.6)
            fig = trend_line(fig, sleep_d["SleepHours"], sleep_d["TotalSteps"])
            fig.update_layout(title="Sleep Hours vs. Same-day Steps", xaxis_title="Sleep (hours)",
                               yaxis_title="Total Steps", legend=dict(orientation="h", y=1.15))
            show(fig)
        else:
            st.info("No sleep records in the current filter selection.")

    section("Weight logging", "Sparse by design — most participants never logged weight")
    left2, right2 = st.columns(2)
    with left2:
        if len(weight_d):
            fig = px.scatter(weight_d, x="Date", y="WeightKg", color=weight_d["Id"].astype(str))
            fig.update_layout(title="Logged Weight Over Time (by participant)", legend_title="Id")
            show(fig)
        else:
            st.info("No weight records in the current filter selection.")
    with right2:
        log_counts = d.groupby("Id", observed=True)["WeightKg"].apply(lambda s: s.notna().sum())
        log_counts = log_counts[log_counts > 0].sort_values(ascending=False)
        if len(log_counts):
            fig = go.Figure(go.Bar(x=log_counts.index.astype(str), y=log_counts.values, marker_color=PURPLE))
            fig.update_layout(title="Weight-log Entries per Participant (logged only)", yaxis_title="Entries")
            show(fig)
        else:
            st.info("No weight records in the current filter selection.")

# ================================================================ 4. SQL Analysis
with tabs[3]:
    st.caption("Queries run against an in-memory SQLite copy of the merged `daily` and `hourly` tables "
               "(full, unfiltered data — sidebar filters don't apply here).")
    con = sql_db()
    labels = [q[0] for q in QUERIES]
    pick = st.selectbox("Pre-written query", labels)
    qdef = next(q for q in QUERIES if q[0] == pick)
    st.caption(qdef[1])
    sql_text = st.text_area("SQL", qdef[2], height=170)
    if st.button("Run query", type="primary"):
        try:
            result = pd.read_sql_query(sql_text, con)
            table(result)
            st.caption(f"{len(result):,} rows")
        except Exception as e:
            st.error(f"Query failed: {e}")

# ================================================================ 5. Recommendations
with tabs[4]:
    st.subheader("Customer-facing recommendations")
    recs = [
        ("Sedentary-break nudges", "Highest impact",
         "Participants average roughly 16+ sedentary hours/day, far outweighing active time. "
         "A move reminder after a set period of inactivity addresses the single biggest pattern in the data."),
        ("Evening engagement window", "Timing",
         "Calorie burn peaks between ~5-7 PM. Scheduling nudges, challenges or notifications around this window "
         "will land when users are already active, instead of competing with an inactive part of the day."),
        ("Personalize over generalize", "Segmentation",
         "Individual variation in steps and calories is far larger than variation across weekdays. "
         "Per-user rolling-average goals will resonate more than a single company-wide weekday campaign."),
        ("Close the sleep gap", "Wellness",
         "Average sleep is a little under 7 hours a night, below the commonly cited 7-9 hour guideline. "
         "A wind-down reminder for users trending under 7 hours could help."),
        ("Revive weight tracking", "Feature adoption",
         "Only a minority of participants log weight at all, and manual entries outnumber automatic syncs. "
         "Prompting automatic weight-log syncing could meaningfully raise this very low engagement rate."),
    ]
    for title, tag, body in recs:
        st.markdown(f"""<div class="rec"><span class="tag">{tag}</span><h4>{title}</h4><p>{body}</p></div>""",
                    unsafe_allow_html=True)

# ================================================================ 6. Data & Methodology
with tabs[5]:
    st.subheader("Data & methodology")
    st.markdown("""
**Source:** Fitbit fitness-tracker export (public, CC0), 33 participants, 12 Apr – 12 May 2016,
distributed via Amazon Mechanical Turk — 19 raw CSV/XLSX files covering daily, hourly, minute
(narrow & wide) and per-second granularity, plus sleep and weight logs.

**Merge approach:** the 19 raw files were combined into 5 files by matching time grain
(`daily_merged`, `hourly_merged`, `minute_narrow_merged`, `minute_wide_merged`,
`heartrate_seconds_merged`), using **outer joins on `Id` + timestamp** so no row from any source
file was dropped. Values were **not corrected, cleaned, or recalculated** — only combined. This
dashboard is built on `daily_merged` and `hourly_merged`; the finer-grained minute and per-second
files are kept out of the dashboard because of their size (GitHub's file-size limits).

**Known data caveats (kept as-is, not corrected):**
- Sleep logs exist for only a subset of participant-days.
- Weight logs are sparse — most participants never logged weight, and manual entries outnumber
  automatic syncs.
- Some participant-days show implausibly low or zero step/calorie totals, consistent with the
  tracker not being worn that day — left in the data rather than filtered out, so any inclusion/
  exclusion decision stays with the analyst using the sidebar filters.

**Tools used:** pandas for merging & aggregation \u00b7 SQLite for the SQL Lab \u00b7 Plotly for the
interactive dashboard \u00b7 Streamlit for the app shell (this file).
""")
    st.caption("Full column list — daily_merged:")
    st.code(", ".join(daily_all.columns), language="text")
    st.caption("Full column list — hourly_merged:")
    st.code(", ".join(hourly_all.columns), language="text")
