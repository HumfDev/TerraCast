"""
Weather-to-Yield Signal Detection Dashboard — Databricks
---------------------------------------------------------
Same as weather_yield_dashboard.py but supports:
  - Spark table (when running in Databricks: notebook "Run as Streamlit" or App on cluster)
  - CSV path (local or when no Spark)

In Databricks:
  1. Put merged data in a table: spark.table("catalog.schema.merged_yield_weather") or default.merged_yield_weather
  2. Create an App (Streamlit) with this file as entry point, or run notebook with this code and "Run as Streamlit app"
  3. In sidebar choose "Spark table" and enter the table name.

Run locally: streamlit run weather_yield_dashboard_databricks.py
Requirements: streamlit pandas numpy scikit-learn plotly
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Spark: available in Databricks notebook "Run as Streamlit"; not available in deployed Apps (use SQL Warehouse instead).
try:
    spark = globals().get("spark", None)
except NameError:
    spark = None

# --- Core message ---
CORE_MSG = (
    "Yield variation is driven more by when, how long, and how extreme "
    "weather stress occurs than by average temperature."
)

DRIVER_GROUPS = {
    "Heat stress": [
        "heat_stress_days", "peak_temp_max_c", "avg_temp_max_c",
        "extreme_heat_flag", "gdd", "avg_temp_mean_c",
    ],
    "Water availability": [
        "avg_precip_mm", "dry_days", "drought_flag", "flood_flag",
        "winter_snowfall_mm",
    ],
    "Timing / variability": [
        "heavy_rain_days", "precip_std_mm", "frost_days",
    ],
}

DEFAULT_CSV = "workspace.default.merged"
WEATHER_FEATURES = [
    "avg_precip_mm", "dry_days", "heavy_rain_days", "precip_std_mm",
    "avg_temp_max_c", "avg_temp_min_c", "avg_temp_mean_c", "peak_temp_max_c",
    "heat_stress_days", "gdd", "frost_days", "winter_snowfall_mm",
    "drought_flag", "flood_flag", "extreme_heat_flag",
]
TARGET = "yield_amount"

STATE_CENTROIDS = {
    "AL": (32.9, -86.9), "AR": (34.9, -92.4), "IA": (41.9, -93.6), "IL": (40.0, -89.0), "IN": (40.3, -86.1),
    "KS": (38.5, -98.5), "KY": (37.5, -85.3), "MD": (39.0, -76.6), "MI": (43.3, -84.5), "MN": (46.0, -94.0),
    "MO": (37.9, -91.5), "NC": (35.6, -79.4), "ND": (47.5, -100.5), "NE": (41.1, -98.0), "OH": (40.4, -82.8),
    "PA": (40.9, -77.6), "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.6), "TX": (31.0, -100.0),
    "WI": (44.5, -89.6), "NY": (43.0, -75.5), "CO": (39.1, -105.3),
}


def _normalize_and_filter(df: pd.DataFrame, commodity_filter: str | None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    if commodity_filter:
        df = df[df["commodity"].astype(str).str.strip().str.lower() == commodity_filter.strip().lower()]
    return df


@st.cache_data
def load_data_csv(path: str, commodity_filter: str | None):
    df = pd.read_csv(path)
    return _normalize_and_filter(df, commodity_filter)


@st.cache_data
def load_data_spark(table_name: str, commodity_filter: str | None):
    if spark is None:
        raise RuntimeError("Spark is not available. Use CSV or SQL Warehouse, or run from Databricks notebook.")
    sdf = spark.table(table_name.strip())
    df = sdf.toPandas()
    return _normalize_and_filter(df, commodity_filter)


@st.cache_data
def load_data_sql_warehouse(server_hostname: str, http_path: str, table_name: str, commodity_filter: str | None):
    """Load from Unity Catalog via Databricks SQL Connector. Token from env DATABRICKS_ACCESS_TOKEN or DATABRICKS_TOKEN."""
    try:
        from databricks import sql as databricks_sql
    except ImportError:
        raise RuntimeError("Install databricks-sql-connector for SQL Warehouse source: pip install databricks-sql-connector")
    token = os.environ.get("DATABRICKS_ACCESS_TOKEN") or os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError(
            "Set access token via environment variable: DATABRICKS_ACCESS_TOKEN or DATABRICKS_TOKEN. "
            "Do not put the token in code."
        )
    conn = databricks_sql.connect(
        server_hostname=server_hostname.strip(),
        http_path=http_path.strip(),
        access_token=token,
    )
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table_name.strip()}")
        df = cursor.fetchall_arrow().to_pandas()
    conn.close()
    return _normalize_and_filter(df, commodity_filter)


def get_available_features(df):
    out = []
    for c in WEATHER_FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 100:
            continue
        out.append(c)
    return out


def feature_to_driver_group(feature: str) -> str:
    for group_name, feats in DRIVER_GROUPS.items():
        if feature in feats:
            return group_name
    return "Other"


def feature_importance_correlation(df, features):
    corrs = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        r = df[TARGET].corr(s)
        corrs.append({"feature": f, "correlation": round(r, 4), "abs_correlation": abs(r)})
    corrs.sort(key=lambda x: -x["abs_correlation"])
    return pd.DataFrame(corrs)


def feature_importance_model(df, features):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].values
    mask = np.isfinite(y)
    X, y = X.loc[mask], y[mask]
    if len(X) < 50:
        return pd.DataFrame([{"feature": f, "importance": 0.0} for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def flag_anomalies(df, features):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].values
    mask = np.isfinite(y)
    if mask.sum() < 50:
        df = df.copy()
        df["yield_predicted"] = np.nan
        df["yield_residual"] = np.nan
        df["anomaly_flag"] = ""
        return df
    X_fit, y_fit = X.loc[mask], y[mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_fit)
    pred = model.predict(X_scaled)
    residual = y_fit - pred
    thresh = 2.0 * np.nanstd(residual) if np.nanstd(residual) > 0 else 0
    df = df.copy()
    df["yield_predicted"] = np.nan
    df["yield_residual"] = np.nan
    df.loc[mask, "yield_predicted"] = pred
    df.loc[mask, "yield_residual"] = residual
    df["anomaly_flag"] = ""
    df.loc[mask, "anomaly_flag"] = np.where(
        df.loc[mask, "yield_residual"] > thresh, "high_yield_vs_weather",
        np.where(df.loc[mask, "yield_residual"] < -thresh, "low_yield_vs_weather", ""),
    )
    return df


def add_zscore_anomaly(df, group_cols=None):
    if group_cols is None:
        group_cols = [c for c in ["state_fips", "county_fips", "commodity"] if c in df.columns]
    if not group_cols:
        df["yield_zscore"] = np.nan
        df["anomaly_flag_zscore"] = ""
        return df
    df = df.copy()
    grp = df.groupby(group_cols)[TARGET].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan)
    df["yield_zscore"] = grp
    df["anomaly_flag_zscore"] = ""
    z = df["yield_zscore"]
    df.loc[z > 2, "anomaly_flag_zscore"] = "high_z"
    df.loc[z < -2, "anomaly_flag_zscore"] = "low_z"
    return df


def build_explainer(corr_rank, model_rank, df):
    lines = [
        "**Core message**",
        f"> {CORE_MSG}",
        "",
        "---",
        "### Top drivers (3 groups)",
        "",
        "| Group | Representative variables (top by correlation/importance) |",
        "|-------|--------------------------------------------------------|",
    ]
    for group_name in ["Heat stress", "Water availability", "Timing / variability"]:
        feats = [f for f in model_rank["feature"] if feature_to_driver_group(f) == group_name]
        feats = feats[:3] or ["-"]
        lines.append(f"| **{group_name}** | {', '.join(feats)} |")
    n_high = (df["anomaly_flag"] == "high_yield_vs_weather").sum()
    n_low = (df["anomaly_flag"] == "low_yield_vs_weather").sum()
    lines.extend([
        "", "### Anomalies (Residual)",
        f"- **Higher yield** than weather suggests: {n_high} — possible protective factors (irrigation, soil, variety, management)",
        f"- **Lower yield** than weather suggests: {n_low} — possible: pests, reporting delay, data gaps, local disasters",
        "", "Extreme events: drought_flag, flood_flag, extreme_heat_flag (e.g. 2012/2021 drought/heatwave).",
    ])
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Weather–Yield Signal", layout="wide")
    st.title("Weather-to-Yield Signal Detection Dashboard")

    st.info(f"**Core message** — {CORE_MSG}")

    # Sidebar: data source
    st.sidebar.header("Settings")
    use_spark = spark is not None
    sources = ["CSV file", "SQL Warehouse (UC table)"]
    if use_spark:
        sources.insert(0, "Spark table")
    data_source = st.sidebar.radio("Data source", sources, horizontal=True)

    table_name = None
    sql_server_hostname = None
    sql_http_path = None
    csv_path = None
    if data_source == "Spark table":
        table_name = st.sidebar.text_input(
            "Spark / UC table name",
            value="merged_yield_weather",
            help="e.g. catalog.schema.merged_yield_weather",
        )
    elif data_source == "SQL Warehouse (UC table)":
        sql_server_hostname = st.sidebar.text_input(
            "Server hostname",
            value=os.environ.get("DATABRICKS_HOST", "dbc-72018123-58b0.cloud.databricks.com"),
            help="e.g. dbc-xxxxx.cloud.databricks.com",
        )
        sql_http_path = st.sidebar.text_input(
            "SQL Warehouse HTTP path",
            value=os.environ.get("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/b2bb07abff382b79"),
            help="From SQL Warehouse → Connection details",
        )
        table_name = st.sidebar.text_input("UC table name", value="catalog.schema.merged_yield_weather")
        st.sidebar.caption("Token: set env DATABRICKS_ACCESS_TOKEN (do not paste in app)")
    else:
        csv_path = st.sidebar.text_input("Merged CSV path", value=str(DEFAULT_CSV))

    commodity_filter = st.sidebar.selectbox("Commodity filter", [None, "Corn", "Soybeans"], format_func=lambda x: "All" if x is None else x)

    # Load data
    if data_source == "Spark table" and table_name:
        try:
            with st.spinner("Loading from Spark table..."):
                df = load_data_spark(table_name, commodity_filter)
        except Exception as e:
            st.error(f"Failed to load table: {e}")
            return
    elif data_source == "SQL Warehouse (UC table)" and sql_server_hostname and sql_http_path and table_name:
        try:
            with st.spinner("Loading from SQL Warehouse..."):
                df = load_data_sql_warehouse(sql_server_hostname, sql_http_path, table_name, commodity_filter)
        except Exception as e:
            st.error(f"Failed to load from SQL Warehouse: {e}")
            return
    elif data_source == "CSV file" and csv_path:
        if not Path(csv_path).exists():
            st.error(f"File not found: {csv_path}")
            return
        with st.spinner("Loading data..."):
            df = load_data_csv(csv_path, commodity_filter)
    else:
        st.warning("Choose data source and fill in table name / HTTP path / CSV path.")
        return

    features = get_available_features(df)
    if not features:
        st.error("No weather features found in the data.")
        return

    st.sidebar.metric("Rows", len(df))
    st.sidebar.metric("Weather features", len(features))

    corr_rank = feature_importance_correlation(df, features)
    model_rank = feature_importance_model(df, features)
    df = flag_anomalies(df, features)
    df = add_zscore_anomaly(df)

    corr_rank = corr_rank.copy()
    corr_rank["driver_group"] = corr_rank["feature"].map(feature_to_driver_group)
    model_rank = model_rank.copy()
    model_rank["driver_group"] = model_rank["feature"].map(feature_to_driver_group)

    def key_display_cols(d):
        return [c for c in ["county_name", "state_abbr", "year", "commodity", "yield_amount", "yield_predicted", "yield_residual", "heat_stress_days", "dry_days", "avg_precip_mm"] if c in d.columns]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Overview", "2. Feature Importance (driver groups)", "3. Anomalies",
        "4. Map + Top 10 cards", "5. Case Studies", "6. What Drives Yield?",
    ])

    with tab1:
        st.caption("Slide 1 — Problem & Why it matters")
        st.markdown("> **Yield is determined once a year, but its drivers accumulate daily as weather stress.**")
        st.markdown("> We summarize weather at county-year level using **agronomic-mechanism-based features** to explain yield and detect anomalies.")
        st.markdown(f"*{CORE_MSG}*")
        with st.expander("Slide 2 — Turning Daily Weather into Agronomic Signals (method)"):
            st.markdown("Not simple averages:")
            st.markdown("- **GDD** (cumulative growing degree days) · **Heatwave days** (Tmax>95°F → `heat_stress_days`)")
            st.markdown("- **Longest dry spell** → `dry_days` · **Heavy-rain bursts** (max 5-day rain) → `heavy_rain_days`")
            st.markdown("*Extremes are not removed; they are structured as stress indicators.*")
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total rows", len(df))
        with c2:
            n_res = (df["anomaly_flag"] != "").sum()
            st.metric("Anomalies (Residual)", n_res)
        with c3:
            n_z = (df["anomaly_flag_zscore"] != "").sum()
            st.metric("Anomalies (Z-score)", n_z)
        with c4:
            st.metric("States", df["state_abbr"].nunique() if "state_abbr" in df.columns else "-")
        st.dataframe(df.head(200), use_container_width=True)

    with tab2:
        st.caption("Slide 3 — What Drives Yield? (by driver **groups**, not a variable list)")
        st.markdown(f"*{CORE_MSG}*")
        import plotly.express as px
        top_corr = corr_rank.head(12).copy()
        top_corr["feature_grp"] = top_corr["feature"] + " (" + top_corr["driver_group"] + ")"
        fig1 = px.bar(
            top_corr, x="abs_correlation", y="feature_grp", orientation="h",
            color="driver_group",
            title="|Correlation| with yield — colored by driver group",
        )
        fig1.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(fig1, use_container_width=True)
        top_model = model_rank.head(12).copy()
        top_model["feature_grp"] = top_model["feature"] + " (" + top_model["driver_group"] + ")"
        fig2 = px.bar(
            top_model, x="importance", y="feature_grp", orientation="h",
            color="driver_group",
            title="RF importance — Top drivers: Heat stress / Water availability / Timing·variability",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Three categories**: Heat stress (heat_days, peak_tmax) · Water availability (precip, dry spell) · Timing/variability (rain volatility, bursts)")

    with tab3:
        st.caption("Anomaly table (Residual + Z-score)")
        st.markdown("*Normal: weather stress → yield decrease. Anomaly: excess decrease or increase not explained by weather.*")
        filter_flag = st.multiselect("Filter by anomaly (Residual)", ["high_yield_vs_weather", "low_yield_vs_weather"], default=None)
        filter_z = st.multiselect("Filter by anomaly (Z-score)", ["high_z", "low_z"], default=None)
        show = df.copy()
        if filter_flag:
            show = show[show["anomaly_flag"].isin(filter_flag)]
        if filter_z:
            show = show[show["anomaly_flag_zscore"].isin(filter_z)]
        st.dataframe(show, use_container_width=True)

    with tab4:
        st.caption("Slide 4 — Where does yield not match the weather?")
        st.markdown("**Residual (actual − predicted)**: positive = higher yield than weather suggests; negative = lower.")
        st.markdown(f"*{CORE_MSG}*")
        if "state_abbr" in df.columns:
            state_agg = df.groupby("state_abbr").agg(
                n_rows=(TARGET, "count"),
                n_anomalies_res=("anomaly_flag", lambda s: (s != "").sum()),
                avg_yield=(TARGET, "mean"),
                avg_residual=("yield_residual", "mean"),
            ).reset_index()
            state_agg["lat"] = state_agg["state_abbr"].map(lambda s: STATE_CENTROIDS.get(str(s).strip(), (None, None))[0])
            state_agg["lon"] = state_agg["state_abbr"].map(lambda s: STATE_CENTROIDS.get(str(s).strip(), (None, None))[1])
            map_df = state_agg.dropna(subset=["lat", "lon"])
            if not map_df.empty:
                import plotly.express as px
                fig = px.scatter_geo(
                    map_df, lat="lat", lon="lon",
                    color="avg_residual", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    size=np.clip(map_df["n_anomalies_res"] + 1, 2, 25),
                    hover_name="state_abbr",
                    hover_data={"avg_residual": ":.2f", "avg_yield": ":.1f", "n_anomalies_res": True, "lat": False, "lon": False},
                    scope="usa", title="State-level average residual (color): positive = higher yield than weather suggests",
                )
                fig.update_geos(showcoastlines=True)
                st.plotly_chart(fig, use_container_width=True)
            res = df.dropna(subset=["yield_residual"]).copy()
            res["abs_residual"] = res["yield_residual"].abs()
            top10 = res.nlargest(10, "abs_residual")
            st.subheader("Top 10 anomaly county-year cards")
            key_cols = [c for c in ["county_name", "state_abbr", "year", "commodity", "yield_amount", "yield_predicted", "yield_residual"] if c in df.columns]
            stress_cols = [c for c in ["heat_stress_days", "dry_days", "avg_precip_mm", "drought_flag", "extreme_heat_flag"] if c in df.columns]
            for i, (_, row) in enumerate(top10.iterrows(), 1):
                with st.expander(f"#{i} {row.get('county_name', '')} {row.get('state_abbr', '')} {row.get('year', '')} — residual {row.get('yield_residual', 0):.2f}"):
                    def _serialize(v):
                        if pd.isna(v): return None
                        if isinstance(v, (np.floating, np.integer)): return float(v)
                        return str(v)
                    st.json({k: _serialize(v) for k, v in row[key_cols + stress_cols].to_dict().items()})
        else:
            st.warning("No state_abbr column for map.")

    with tab5:
        st.caption("Slide 5 — Explain 2–3 case studies (highlight of the story)")
        st.markdown(f"*{CORE_MSG}*")
        st.subheader("Case A: High yield despite heat/drought")
        case_a = df[(df["anomaly_flag"] == "high_yield_vs_weather")]
        st.markdown("→ **Inference**: possible **protective factors** (irrigation, soil, variety, management).")
        if len(case_a) > 0:
            st.dataframe(case_a.head(20)[key_display_cols(df)], use_container_width=True)
        else:
            st.caption("No cases (depends on filter/data)")
        st.divider()
        st.subheader("Case B: Yield drop despite favorable weather")
        case_b = df[(df["anomaly_flag"] == "low_yield_vs_weather")]
        st.markdown("→ **Possible**: pests, reporting delay, data gaps, local disasters.")
        if len(case_b) > 0:
            st.dataframe(case_b.head(20)[key_display_cols(df)], use_container_width=True)
        else:
            st.caption("No cases")
        st.divider()
        st.subheader("Case C: Selected year (drought/heat) — heat_days vs normal, predicted vs actual")
        years = sorted(df["year"].dropna().unique()) if "year" in df.columns else []
        year_opts = [y for y in years if y >= 2015] if years else []
        case_year = st.selectbox("Analysis year", year_opts, index=len(year_opts) - 1 if year_opts else 0) if year_opts else None
        if case_year is not None and "heat_stress_days" in df.columns and "year" in df.columns:
            full = df.dropna(subset=["yield_residual"])
            yr_df = full[full["year"] == case_year]
            other = full[full["year"] != case_year]
            avg_heat_other = other["heat_stress_days"].mean()
            avg_heat_yr = yr_df["heat_stress_days"].mean()
            st.metric(f"{case_year} avg heat_stress_days", f"{avg_heat_yr:.1f}")
            st.metric("Other years average", f"{avg_heat_other:.1f}")
            st.markdown(f"→ This year heat_days is **{avg_heat_yr - avg_heat_other:+.1f}** days vs normal. Model predicted yield decrease and actual decreased—but some counties show mismatch (anomaly).")
            mismatch = yr_df[yr_df["anomaly_flag"] != ""]
            st.dataframe(mismatch.head(15)[key_display_cols(df)], use_container_width=True)
        else:
            st.caption("No year or heat_stress_days in data")

    with tab6:
        st.markdown(build_explainer(corr_rank, model_rank, df))


if __name__ == "__main__":
    main()
