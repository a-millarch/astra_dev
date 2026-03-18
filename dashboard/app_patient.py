# requires streamlit

import streamlit as st
import pandas as pd
from cstar import ProjectManager
import cstar.azure as az
import plotly.graph_objects as go
import streamlit.components.v1 as components
import numpy as np

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
pm = ProjectManager()

from astra.utils import get_base_df
from astra.data.filters import filter_vitals, filter_labs, filter_ita, filter_medicin, filter_procedures, filter_adt
from astra.data.notes_features import build_gcs_from_notes, build_iss_from_notes, build_intubation_from_notes
from astra.data.cardiac_arrest import build_cardiac_arrest_from_notes

@st.cache_data(show_spinner="Indlæser base-data …")
def load_data() -> pd.DataFrame:
    return get_base_df()

df = load_data()


# ── Sideopsætning ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Lookup – ASTRA",
    layout="wide",
)

# ── Sidebar: vælg patient ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Vælg patient")

    # Fritekst-søgning
    search_input = st.text_input("Søg PID", placeholder="Skriv PID her …")

    # Filtrer liste baseret på søgning
    all_pids = df["PID"].astype(str).unique().tolist()
    filtered_pids = (
        [p for p in all_pids if search_input in p]
        if search_input
        else all_pids
    )

    selected_pid = st.selectbox(
        "Eller vælg fra liste",
        options=filtered_pids,
        index=0 if filtered_pids else None,
    )

    st.markdown("---")
    st.caption(f"Antal patienter i datasættet: **{len(all_pids):,}**")

# ── Hent valgt patient ────────────────────────────────────────────────────────
if selected_pid is None:
    st.warning("Ingen patient valgt.")
    st.stop()

patient_row = df[df["PID"].astype(str) == str(selected_pid)]

if patient_row.empty:
    st.error(f"PID **{selected_pid}** blev ikke fundet i datasættet.")
    st.stop()

# ── Overblik: base ─────────────────────────────────────────────────────
row = patient_row.iloc[0].copy()
for col in patient_row.select_dtypes(include=["timedelta"]).columns:
    row[col] = str(row[col])

# Rens datoer og afrund tal
for col in patient_row.select_dtypes(include=["datetime"]).columns:
    val = row[col]
    if pd.notna(val):
        row[col] = val.strftime("%Y-%m-%d") if val.hour == 0 and val.minute == 0 else val.strftime("%Y-%m-%d %H:%M")

if "DURATION" in row.index and pd.notna(row["DURATION"]):
    row["DURATION"] = round(float(row["DURATION"]), 2)

if "HEIGHT" in row.index and pd.notna(row["HEIGHT"]):
    row["HEIGHT"] = round(float(row["HEIGHT"]), 2)

if "WEIGHT" in row.index and pd.notna(row["WEIGHT"]):
    row["WEIGHT"] = round(float(row["WEIGHT"]), 2)

DEMOGRAFI      = ["SEX", "AGE", "HEIGHT", "WEIGHT", "ASMT_ELIX", "deceased_30d", "deceased_90d"]
FORLØB_VENSTRE = ["ServiceDate", "start", "end", "DURATION", "DOD"]
FORLØB_HØJRE   = ["first_afsnit", "first_RH", "time_to_RH", "type_visitation", "FIRST_HOSPITAL"]

# ── Patient header + demografi ────────────────────────────────────────────────
demo_items = [(c, row[c]) for c in DEMOGRAFI if c in row.index]
demo_html = "&emsp;".join(
    f"<span style='color:#aaa; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em'>{k}</span>"
    f"&ensp;<span style='color:#222; font-size:0.88rem'>{v}</span>"
    for k, v in demo_items
)
st.markdown(
    f"""<div style="padding:0 0 14px 0; border-bottom:1px solid #e5e5e5; margin-bottom:20px">
        <span style="font-size:1.1rem; font-weight:700; color:#111; margin-right:20px">Patient {selected_pid}</span><br><br>
        {demo_html}
    </div>""",
    unsafe_allow_html=True
)

# ── Forløb ────────────────────────────────────────────────────────────────────
def tabel_html(items):
    rows = "".join(
        f"""<tr>
              <td style="color:#aaa; font-size:0.78rem; text-transform:uppercase;
                         letter-spacing:0.04em; padding:6px 24px 6px 0; white-space:nowrap">{k}</td>
              <td style="color:#222; font-size:0.88rem; padding:6px 0;
                         border-bottom:1px solid #f0f0f0">{v}</td>
            </tr>"""
        for k, v in items
    )
    return f"<table style='border-collapse:collapse; width:100%'>{rows}</table>"

v_items = [(c, row[c]) for c in FORLØB_VENSTRE if c in row.index]
h_items = [(c, row[c]) for c in FORLØB_HØJRE   if c in row.index]

left, right = st.columns(2)
with left:
    st.markdown(
        f"""<div style="padding-right:40px">
            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
                        color:#aaa; margin-bottom:10px">Forløb</div>
            {tabel_html(v_items)}
        </div>""",
        unsafe_allow_html=True
    )
with right:
    st.markdown(
        f"""<div style="border-left:1px solid #e5e5e5; padding-left:40px">
            {tabel_html(h_items)}
        </div>""",
        unsafe_allow_html=True

        
    )


@st.cache_data(show_spinner="Indlæser vitale værdier …")
def load_vitals():
    return pd.read_pickle("data/interim/concepts/VitaleVaerdier.pkl")

@st.cache_data(show_spinner="Indlæser laboratoriesvar …")
def load_labs():
    return pd.read_pickle("data/interim/concepts/Labsvar.pkl")

@st.cache_data(show_spinner="Indlæser ICU-målinger …")
def load_icu():
    return pd.read_pickle("data/interim/concepts/ITAOversigtsrapport.pkl")

@st.cache_data(show_spinner="Indlæser medicin …")
def load_medicin():
    return pd.read_pickle("data/interim/concepts/Medicin.pkl")

@st.cache_data(show_spinner="Indlæser procedurer …")
def load_procedurer():
    return pd.read_pickle("data/interim/concepts/Procedurer.pkl")

@st.cache_data(show_spinner="Indlæser ADT …")
def load_adt():
    return pd.read_pickle("data/interim/concepts/ADTHaendelser.pkl")

@st.cache_data(show_spinner="Indlæser EWS …")
def load_ews():
    return pd.read_pickle("data/interim/concepts/EWS.pkl")

@st.cache_data(show_spinner="Indlæser diagnoser …")
def load_diagnoser():
    return pd.read_pickle("data/interim/concepts/Diagnoser.pkl")

@st.cache_data(show_spinner="Indlæser notater …")
def load_notater():
    return pd.read_pickle("data/interim/concepts/Notater.pkl")


# ── Per-patient cached data ──────────────────────────────────────────────────
# These functions filter + process heavy concept data ONCE per patient.
# Subsequent reruns (e.g. timeframe changes) skip the expensive work.

@st.cache_data(show_spinner="Filtrerer vitale værdier for patient …")
def get_patient_vitals(pid: int):
    vitals_raw = load_vitals()
    vitals_pid = vitals_raw[vitals_raw["PID"] == pid].copy()
    if vitals_pid.empty:
        return vitals_pid
    ews_raw = load_ews()
    ews_pid = ews_raw[ews_raw["PID"] == pid].copy()
    vitals_pid = filter_vitals(vitals_pid, ews=ews_pid if not ews_pid.empty else None)
    vitals_pid["VALUE"] = pd.to_numeric(vitals_pid["VALUE"], errors="coerce")
    return vitals_pid

@st.cache_data(show_spinner="Filtrerer laboratoriesvar for patient …")
def get_patient_labs(pid: int):
    labs_raw = load_labs()
    labs_pid = labs_raw[labs_raw["PID"] == pid].copy()
    if labs_pid.empty:
        return labs_pid
    labs_pid = filter_labs(labs_pid)
    return labs_pid

@st.cache_data(show_spinner="Filtrerer ICU-målinger for patient …")
def get_patient_icu(pid: int):
    icu_raw = load_icu()
    icu_pid = icu_raw[icu_raw["PID"] == pid].copy()
    gcs_notes = _get_gcs_from_notes()
    gcs_notes_pid = gcs_notes[gcs_notes["PID"] == pid].copy()

    if icu_pid.empty and gcs_notes_pid.empty:
        return pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE", "Kilde"])

    if not icu_pid.empty:
        icu_pid = filter_ita(icu_pid)
        icu_pid["Kilde"] = "Rådata"
    else:
        icu_pid = pd.DataFrame(columns=["PID", "TIMESTAMP", "FEATURE", "VALUE", "Kilde"])

    if not gcs_notes_pid.empty:
        gcs_notes_pid["Kilde"] = "Notater"
        icu_pid = pd.concat([icu_pid, gcs_notes_pid], ignore_index=True)

    icu_pid["VALUE"] = pd.to_numeric(icu_pid["VALUE"], errors="coerce")
    return icu_pid

@st.cache_data(show_spinner="Filtrerer medicin for patient …")
def get_patient_medicin(pid: int):
    med_raw = load_medicin()
    med_pid = med_raw[med_raw["PID"] == pid].copy()
    if med_pid.empty:
        return med_pid
    med_pid = filter_medicin(med_pid)
    med_pid["TIMESTAMP"] = pd.to_datetime(med_pid["TIMESTAMP"])
    return med_pid

@st.cache_data(show_spinner="Filtrerer procedurer for patient …")
def get_patient_procedurer(pid: int):
    proc_raw = load_procedurer()
    proc_pid = proc_raw[proc_raw["PID"] == pid].copy()
    if proc_pid.empty:
        return proc_pid
    proc_pid_filtered = filter_procedures(proc_pid)
    proc_pid_filtered["TIMESTAMP"] = pd.to_datetime(proc_pid_filtered["TIMESTAMP"])
    return proc_pid_filtered

@st.cache_data(show_spinner="Filtrerer procedurer (alle) for patient …")
def get_patient_procedurer_all(pid: int):
    proc_raw = load_procedurer()
    proc_pid = proc_raw[proc_raw["PID"] == pid].copy()
    if proc_pid.empty:
        return proc_pid
    proc_pid["ServiceDatetime"] = pd.to_datetime(proc_pid["ServiceDatetime"], errors="coerce")
    proc_pid = proc_pid.dropna(subset=["ServiceDatetime"]).sort_values("ServiceDatetime")
    return proc_pid

@st.cache_data(show_spinner="Filtrerer ADT for patient …")
def get_patient_adt(pid: int, _base_df):
    adt_raw = load_adt()
    adt_pid = adt_raw[adt_raw["PID"] == pid].copy()
    if adt_pid.empty:
        return adt_pid
    adt_filtered = filter_adt(adt_pid, base_df=_base_df)
    adt_filtered["TIMESTAMP"] = pd.to_datetime(adt_filtered["TIMESTAMP"])
    adt_filtered["END_TIMESTAMP"] = pd.to_datetime(adt_filtered["END_TIMESTAMP"])
    return adt_filtered

@st.cache_data(show_spinner="Filtrerer ADT (alle) for patient …")
def get_patient_adt_all(pid: int):
    adt_raw = load_adt()
    adt_pid = adt_raw[adt_raw["PID"] == pid].copy()
    if adt_pid.empty:
        return adt_pid
    adt_pid["Flyt_ind"] = pd.to_datetime(adt_pid["Flyt_ind"], errors="coerce")
    adt_pid["Flyt_ud"] = pd.to_datetime(adt_pid["Flyt_ud"], errors="coerce")
    adt_pid = adt_pid.dropna(subset=["Flyt_ind"]).sort_values("Flyt_ind")
    adt_pid = adt_pid[adt_pid["Flyt_ind"] != adt_pid["Flyt_ud"]].copy()
    # Merge consecutive stays at same ward
    adt_pid = adt_pid.sort_values("Flyt_ind").reset_index(drop=True)
    merged_rows = []
    for _, r in adt_pid.iterrows():
        if merged_rows and (
            merged_rows[-1]["Afsnit"] == r["Afsnit"] and
            merged_rows[-1]["Flyt_ud"] == r["Flyt_ind"]
        ):
            merged_rows[-1]["Flyt_ud"] = r["Flyt_ud"]
        else:
            merged_rows.append(r.to_dict())
    return pd.DataFrame(merged_rows) if merged_rows else pd.DataFrame()

@st.cache_data(show_spinner="Filtrerer diagnoser for patient …")
def get_patient_diagnoser(pid: int):
    diag_raw = load_diagnoser()
    diag_pid = diag_raw[diag_raw["PID"] == pid].copy()
    if diag_pid.empty:
        return diag_pid
    diag_pid["Noteret_dato"] = pd.to_datetime(diag_pid["Noteret_dato"], errors="coerce")
    diag_pid = diag_pid.dropna(subset=["Noteret_dato"]).sort_values("Noteret_dato")
    return diag_pid

@st.cache_data(show_spinner="Filtrerer notater for patient …")
def get_patient_notater(pid: int):
    noter_raw = load_notater()
    noter_pid = noter_raw[noter_raw["PID"] == pid].copy()
    if noter_pid.empty:
        return noter_pid
    noter_pid = noter_pid.sort_values(["ID", "Linjenummer"])
    noter_pid = (
        noter_pid.groupby(["ID", "Oprettelsestidspunkt", "Redigeringstidspunkt",
                           "Notetype", "Speciale"], as_index=False, dropna=False)
        .agg({"Note": lambda x: " ".join(x.astype(str))})
    )
    noter_pid["Oprettelsestidspunkt"] = pd.to_datetime(noter_pid["Oprettelsestidspunkt"])
    noter_pid["Note"] = noter_pid["Note"].str.replace("    ", "\n\n").str.replace("  ", "\n")
    noter_pid = noter_pid.sort_values("Oprettelsestidspunkt", ascending=True).reset_index(drop=True)
    return noter_pid

# Cache the expensive notes-derived features globally (computed once for ALL patients)
@st.cache_data(show_spinner="Bygger GCS fra notater …")
def _get_gcs_from_notes():
    return build_gcs_from_notes(load_notater())

@st.cache_data(show_spinner="Bygger ISS fra notater …")
def _get_iss_from_notes():
    return build_iss_from_notes(load_notater())

@st.cache_data(show_spinner="Bygger intubation fra notater …")
def _get_intubation_from_notes():
    return build_intubation_from_notes(load_notater())

@st.cache_data(show_spinner="Bygger hjertestop fra notater …")
def _get_cardiac_arrest_from_notes():
    return build_cardiac_arrest_from_notes(load_notater())

# ── Tidsvindue i sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**Tidsvindue**")
    patient_start = pd.to_datetime(row["start"])
    patient_end = pd.to_datetime(row["end"])
    forløb_timer = max((patient_end - patient_start).total_seconds() / 3600, 1.0)
    
    tidsvindue = st.slider(
        "Timer fra forløbsstart",
        min_value=0.5,
        max_value=round(forløb_timer, 1),
        value=round(forløb_timer, 1),
        step=0.5,
    )
    tidsvindue = st.number_input(
        "Eller skriv antal timer",
        min_value=0.5,
        max_value=round(forløb_timer, 1),
        value=tidsvindue,
        step=0.5,
    )

patient_start = pd.to_datetime(row["start"])

def tilfoej_timer(df, ts_col):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["timer_siden_start"] = (df[ts_col] - patient_start).dt.total_seconds() / 3600
    return df[
        (df["timer_siden_start"] >= 0) &
        (df["timer_siden_start"] <= tidsvindue)
    ]

PLOT_HEIGHT = 250
ROW_HEIGHT = 35
HEADER_HEIGHT = 38

def vis_feature_rækker(filtered_df, ts_col):
    def format_timer(h):
        if h < 1:
            return f"{int(h * 60)}m"
        elif h < 24:
            t = int(h)
            m = int((h % 1) * 60)
            return f"{t}t {m}m" if m > 0 else f"{t}t"
        else:
            d = int(h // 24)
            t = int(h % 24)
            return f"{d}d {t}t" if t > 0 else f"{d}d"

    features = sorted(filtered_df["FEATURE"].unique())
    for feature in features:
        feat_df = (
            filtered_df[filtered_df["FEATURE"] == feature]
            .sort_values(ts_col)[["timer_siden_start", ts_col, "VALUE"]]
            .dropna(subset=["VALUE"])
            .reset_index(drop=True)
        )
        st.markdown(f"**{feature}**")
        tabel_col, graf_col = st.columns([1, 2])

        with tabel_col:
            if feat_df.empty:
                st.caption("Ingen gyldige værdier")
            else:
                tabel_vis = feat_df.copy()
                tabel_vis[ts_col] = tabel_vis[ts_col].dt.strftime("%Y-%m-%d %H:%M")
                tabel_vis["VALUE"] = tabel_vis["VALUE"].round(2)
                tabel_vis = tabel_vis[[ts_col, "VALUE"]]
                tabel_vis.columns = ["Tidspunkt", "Værdi"]
                dynamisk_højde = min(HEADER_HEIGHT + len(tabel_vis) * ROW_HEIGHT, PLOT_HEIGHT)
                st.dataframe(tabel_vis, hide_index=True, use_container_width=True, height=dynamisk_højde)

        with graf_col:
            if len(feat_df) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=feat_df["timer_siden_start"],
                    y=feat_df["VALUE"],
                    mode="lines",
                    line=dict(color="#4e79a7"),
                    hovertemplate="%{customdata}<br>Værdi: %{y}<extra></extra>",
                    customdata=[format_timer(h) for h in feat_df["timer_siden_start"]],
                ))

                max_timer = feat_df["timer_siden_start"].max()
                min_timer = feat_df["timer_siden_start"].min()

                if max_timer <= 0.5:
                    interval = 0.25      # hvert kvarter
                elif max_timer <= 2:
                    interval = 0.25      # hvert kvarter
                elif max_timer <= 6:
                    interval = 1
                elif max_timer <= 12:
                    interval = 2
                elif max_timer <= 24:
                    interval = 3
                elif max_timer <= 48:
                    interval = 6
                elif max_timer <= 72:
                    interval = 12
                elif max_timer <= 168:
                    interval = 24
                elif max_timer <= 336:
                    interval = 48
                elif max_timer <= 720:
                    interval = 72
                else:
                    interval = 168

                tick_start = int(min_timer // interval) * interval
                # Brug numpy arange for float intervaller
                tick_vals = list(np.arange(tick_start, max_timer + interval, interval))
                tick_vals = [round(v, 4) for v in tick_vals]
                if len(tick_vals) < 2:
                    tick_vals = [min_timer, max_timer]
                tick_text = [format_timer(h) for h in tick_vals]

                x_min = min(min_timer, tick_vals[0]) - interval * 0.1
                x_max = max(max_timer, tick_vals[-1]) + interval * 0.1

                # y-akse range baseret på hele patientens forløb for denne feature
                alle_værdier = filtered_df[filtered_df["FEATURE"] == feature]["VALUE"].dropna()
                y_min = alle_værdier.min() * 0.95
                y_max = alle_værdier.max() * 1.05

                fig.update_layout(
                    height=PLOT_HEIGHT,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(
                        title="Tid i forløb",
                        tickvals=tick_vals,
                        ticktext=tick_text,
                        tickangle=30,
                        showgrid=True,
                        gridcolor="#f0f0f0",
                        range=[x_min, x_max],
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#f0f0f0",
                        range=[y_min, y_max],
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)
            elif len(feat_df) == 1:
                st.metric(label="", value=round(feat_df["VALUE"].iloc[0], 2))

        st.markdown("---")


# ── Faner ─────────────────────────────────────────────────────────────────────
st.markdown("---")

tab_vitals, tab_labs, tab_icu, tab_med, tab_proc, tab_adt, tab_trauma, tab_diag, tab_notater, tab_overview = st.tabs([
    "Vitale værdier", "Laboratoriesvar", "ICU", "Medicin",
    "Procedurer", "Afsnit", "Trauma / Events", "Diagnoser", "Notater", "Summary"
])


with tab_vitals:
    vitals_pid = get_patient_vitals(int(selected_pid))
    if vitals_pid.empty:
        st.info("Ingen vitale værdier for denne patient.")
    else:
        vitals_filtered = tilfoej_timer(vitals_pid, "TIMESTAMP").dropna(subset=["VALUE"])
        if vitals_filtered.empty:
            st.info(f"Ingen målinger indenfor de første {tidsvindue} timer.")
        else:
            vis_feature_rækker(vitals_filtered, "TIMESTAMP")

with tab_labs:
    labs_pid = get_patient_labs(int(selected_pid))
    if labs_pid.empty:
        st.info("Ingen laboratoriesvar for denne patient.")
    else:
        labs_filtered = tilfoej_timer(labs_pid, "TIMESTAMP").dropna(subset=["VALUE"])
        if labs_filtered.empty:
            st.info(f"Ingen laboratoriesvar indenfor de første {tidsvindue} timer.")
        else:
            vis_feature_rækker(labs_filtered, "TIMESTAMP")

with tab_icu:
    icu_pid = get_patient_icu(int(selected_pid))

    if icu_pid.empty:
        st.info("Ingen ICU-målinger for denne patient.")
    else:
        icu_filtered = tilfoej_timer(icu_pid, "TIMESTAMP").dropna(subset=["VALUE"])
        if icu_filtered.empty:
            st.info(f"Ingen ICU-målinger indenfor de første {tidsvindue} timer.")
        else:
            # Show features with source column for GCS
            features = sorted(icu_filtered["FEATURE"].unique())
            for feature in features:
                feat_df = (
                    icu_filtered[icu_filtered["FEATURE"] == feature]
                    .sort_values("TIMESTAMP")[["timer_siden_start", "TIMESTAMP", "VALUE", "Kilde"]]
                    .dropna(subset=["VALUE"])
                    .reset_index(drop=True)
                )
                st.markdown(f"**{feature}**")
                tabel_col, graf_col = st.columns([1, 2])

                with tabel_col:
                    if feat_df.empty:
                        st.caption("Ingen gyldige værdier")
                    else:
                        tabel_vis = feat_df.copy()
                        tabel_vis["TIMESTAMP"] = tabel_vis["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
                        tabel_vis["VALUE"] = tabel_vis["VALUE"].round(2)
                        tabel_vis = tabel_vis[["TIMESTAMP", "VALUE", "Kilde"]]
                        tabel_vis.columns = ["Tidspunkt", "Værdi", "Kilde"]
                        dynamisk_højde = min(HEADER_HEIGHT + len(tabel_vis) * ROW_HEIGHT, PLOT_HEIGHT)
                        st.dataframe(tabel_vis, hide_index=True, use_container_width=True, height=dynamisk_højde)

                with graf_col:
                    if len(feat_df) >= 2:
                        fig = go.Figure()
                        # Single combined line sorted by time, markers colored by source
                        sorted_df = feat_df.sort_values("timer_siden_start").reset_index(drop=True)
                        colors_map = {"Rådata": "#4e79a7", "Notater": "#e15759"}
                        marker_colors = [colors_map.get(k, "#76b7b2") for k in sorted_df["Kilde"]]
                        fig.add_trace(go.Scatter(
                            x=sorted_df["timer_siden_start"],
                            y=sorted_df["VALUE"],
                            mode="markers+lines",
                            line=dict(color="#4e79a7"),
                            marker=dict(color=marker_colors, size=8),
                            showlegend=False,
                        ))
                        # Invisible traces for legend
                        for kilde, color in colors_map.items():
                            if kilde in sorted_df["Kilde"].values:
                                fig.add_trace(go.Scatter(
                                    x=[None], y=[None], mode="markers",
                                    marker=dict(color=color, size=8),
                                    name=kilde,
                                ))
                        alle_værdier = feat_df["VALUE"].dropna()
                        y_min = alle_værdier.min() * 0.95
                        y_max = alle_værdier.max() * 1.05
                        fig.update_layout(
                            height=PLOT_HEIGHT,
                            margin=dict(l=0, r=0, t=10, b=10),
                            xaxis=dict(title="Timer i forløb", showgrid=True, gridcolor="#f0f0f0"),
                            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[y_min, y_max]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(feat_df) == 1:
                        st.metric(label="", value=round(feat_df["VALUE"].iloc[0], 2))

                st.markdown("---")

with tab_med:
    med_pid_filtered = get_patient_medicin(int(selected_pid))
    if med_pid_filtered.empty:
        st.info("Ingen medicin registreret for denne patient.")
    else:
        med_filtered = tilfoej_timer(med_pid_filtered, "TIMESTAMP")

        if med_filtered.empty:
            st.info(f"Ingen medicin indenfor de første {tidsvindue} timer.")
        else:
            kategorier = sorted(med_filtered["VALUE"].unique())
         # ── Overblik øverst: antal pr. kategori ──────────────────────────
            st.markdown("**Overblik**")

            for i, kat in enumerate(kategorier):
                antal = len(med_filtered[med_filtered["VALUE"] == kat])
                with st.expander(f"**{kat}** – {antal} administreringer"):
                    kat_df = (
                        med_filtered[med_filtered["VALUE"] == kat]
                        .sort_values("TIMESTAMP")[["TIMESTAMP", "Generisk_navn"]]
                        .reset_index(drop=True)
                    )
                    kat_df["TIMESTAMP"] = kat_df["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
                    kat_df.columns = ["Tidspunkt", "Præparat"]
                    dynamisk_højde = min(HEADER_HEIGHT + len(kat_df) * ROW_HEIGHT, 250)
                    st.dataframe(kat_df, hide_index=True, use_container_width=False, height=dynamisk_højde)

            st.markdown("---")            

            # ── Samlet plot over forløbet ─────────────────────────────────────
            st.markdown("**Medicin over forløbet**")
            FARVER = [
                "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"
            ]
            fig = go.Figure()
            for i, kat in enumerate(kategorier):
                kat_df = med_filtered[med_filtered["VALUE"] == kat].sort_values("TIMESTAMP")
                fig.add_trace(go.Scatter(
                    x=kat_df["TIMESTAMP"],
                    y=[kat] * len(kat_df),
                    mode="markers",
                    name=kat,
                    marker=dict(color=FARVER[i % len(FARVER)], size=10, symbol="line-ns-open", line=dict(width=2)),
                    hovertemplate="%{x}<br>%{customdata}<extra>" + kat + "</extra>",
                    customdata=kat_df["Generisk_navn"].values,
                ))
            fig.update_layout(
                height=80 + len(kategorier) * 60,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(showgrid=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)


with tab_proc:
    tab_proc_klas, tab_proc_alle = st.tabs(["Klassificeret", "Alle"])

    with tab_proc_klas:
        proc_pid_filtered = get_patient_procedurer(int(selected_pid))
        if proc_pid_filtered.empty:
            st.info("Ingen procedurer registreret for denne patient.")
        else:
            proc_filtered = tilfoej_timer(proc_pid_filtered, "TIMESTAMP")
    
            if proc_filtered.empty:
                st.info(f"Ingen procedurer indenfor de første {tidsvindue} timer.")
            else:
                kategorier = sorted(proc_filtered["VALUE"].unique())
    
                # ── Expanders pr. kategori ────────────────────────────────────────
                for i, kat in enumerate(kategorier):
                    antal = len(proc_filtered[proc_filtered["VALUE"] == kat])
                    with st.expander(f"**{kat}** – {antal} procedurer"):
                        kat_df = (
                            proc_filtered[proc_filtered["VALUE"] == kat]
                            .sort_values("TIMESTAMP")[["TIMESTAMP", "ProcedureName"]]
                            .reset_index(drop=True)
                        )
                        kat_df["TIMESTAMP"] = kat_df["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
                        kat_df.columns = ["Tidspunkt", "Procedure"]
                        dynamisk_højde = min(HEADER_HEIGHT + len(kat_df) * ROW_HEIGHT, 250)
                        st.dataframe(kat_df, hide_index=True, use_container_width=False, height=dynamisk_højde)
    
                st.markdown("---")
    
                # ── Plot over forløbet ────────────────────────────────────────────
                st.markdown("**Procedurer over forløbet**")
                FARVER = [
                    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                    "#9c755f", "#bab0ac"
                ]
                fig = go.Figure()
                for i, kat in enumerate(kategorier):
                    kat_df = proc_filtered[proc_filtered["VALUE"] == kat].sort_values("TIMESTAMP")
                    fig.add_trace(go.Scatter(
                        x=kat_df["TIMESTAMP"],
                        y=[kat] * len(kat_df),
                        mode="markers",
                        name=kat,
                        marker=dict(color=FARVER[i % len(FARVER)], size=10, symbol="line-ns-open", line=dict(width=2)),
                        hovertemplate="%{x}<br>%{customdata}<extra>" + kat + "</extra>",
                        customdata=kat_df["ProcedureName"].values,
                    ))
                fig.update_layout(
                    height=80 + len(kategorier) * 60,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_proc_alle:
        proc_pid2 = get_patient_procedurer_all(int(selected_pid))
        if proc_pid2.empty:
            st.info("Ingen procedurer registreret for denne patient.")
        else:
            proc_pid2 = proc_pid2[
                (proc_pid2["ServiceDatetime"] - patient_start).dt.total_seconds() / 3600 <= tidsvindue
            ].copy()
            if proc_pid2.empty:
                st.info("Ingen procedurer indenfor tidsvinduet.")
            else:
                st.markdown("**Alle procedurer over forløbet**")
                tabel = proc_pid2[["ServiceDatetime", "ProcedureName"]].copy()
                tabel["timer_ind"] = (
                    (tabel["ServiceDatetime"] - patient_start).dt.total_seconds() / 3600
                )
    
                def format_tid(h):
                    if h < 24:
                        return f"{int(h)}t {int((h % 1) * 60)}m"
                    else:
                        dage = int(h // 24)
                        timer = int(h % 24)
                        minutter = int((h % 1) * 60)
                        return f"{dage}d {timer}t {minutter}m"
    
                tabel["tid_label"] = tabel["timer_ind"].apply(format_tid)
                tabel["ServiceDatetime_str"] = tabel["ServiceDatetime"].dt.strftime("%Y-%m-%d %H:%M")
    
                grouped = (
                    tabel.groupby(["ServiceDatetime_str", "tid_label"])["ProcedureName"]
                    .apply(list)
                    .reset_index()
                )
    
                # Byg vandret scrollbar HTML
                kolonner_html = ""
                for _, r in grouped.iterrows():
                    badges = "".join([
                        f"<div style='background:#e8f0fe; color:#1a3a6b; font-size:0.72rem; "
                        f"padding:4px 8px; border-radius:10px; margin-bottom:4px'>{p}</div>"
                        for p in r["ProcedureName"]
                    ])
                    kolonner_html += f"""
                        <div style='min-width:180px; max-width:180px; margin-right:12px;
                                    border-left:2px solid #4e79a7; padding-left:10px; flex-shrink:0'>
                            <div style='color:#888; font-size:0.7rem'>{r['ServiceDatetime_str']}</div>
                            <div style='color:#4e79a7; font-size:0.72rem; font-weight:600;
                                        margin-bottom:6px'>{r['tid_label']}</div>
                            {badges}
                        </div>
                    """
    
                components.html(
                    f"""<div style='display:flex; flex-direction:row; overflow-x:auto;
                                   padding:12px 4px 16px 4px; align-items:flex-start;
                                   font-family:sans-serif'>
                        {kolonner_html}
                    </div>""",
                    height=500,
                    scrolling=True,
                )



with tab_adt:
    tab_adt_klas, tab_adt_alle = st.tabs(["Klassificeret", "Alle"])

    with tab_adt_klas:
        adt_filtered = get_patient_adt(int(selected_pid), df)
        if adt_filtered.empty:
            st.info("Ingen afsnitsregistreringer for denne patient.")
        else:
            # ── Overblik: tid brugt pr. afsnitstype ──────────────────────────
            adt_filtered["varighed_timer"] = (
                (adt_filtered["END_TIMESTAMP"] - adt_filtered["TIMESTAMP"])
                .dt.total_seconds() / 3600
            )
            overblik = (
                adt_filtered.groupby("VALUE")["varighed_timer"]
                .sum()
                .reset_index()
                .rename(columns={"VALUE": "Afsnitstype", "varighed_timer": "Timer total"})
                .sort_values("Timer total", ascending=False)
            )
            cols = st.columns(len(overblik))
            for i, r in overblik.iterrows():
                cols[i].metric(r["Afsnitstype"], f"{r['Timer total']:.1f} timer")

            st.markdown("---")

            # ── Gantt-plot ────────────────────────────────────────────────────
            st.markdown("**Patientens forløb gennem afsnit**")

            FARVER = {
                "TC":  "#e15759",
                "OR":  "#4e79a7",
                "ICU": "#f28e2b",
                "BED": "#76b7b2",
                "AMB": "#59a14f",
            }

            fig = go.Figure()
            for _, r in adt_filtered.sort_values("TIMESTAMP").iterrows():
                farve = FARVER.get(r["VALUE"], "#bab0ac")
                fig.add_trace(go.Bar(
                    x=[(r["END_TIMESTAMP"] - r["TIMESTAMP"]).total_seconds() / 3600],
                    y=[r["VALUE"]],
                    base=[(r["TIMESTAMP"] - patient_start).total_seconds() / 3600],
                    orientation="h",
                    marker=dict(color=farve, opacity=0.85),
                    name=r["VALUE"],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{r['VALUE']}</b><br>"
                        f"Ind: {r['TIMESTAMP'].strftime('%Y-%m-%d %H:%M')}<br>"
                        f"Ud: {r['END_TIMESTAMP'].strftime('%Y-%m-%d %H:%M')}<br>"
                        f"Varighed: {(r['END_TIMESTAMP']-r['TIMESTAMP']).total_seconds()/3600:.1f} timer"
                        "<extra></extra>"
                    ),
                ))

            # Én legend-entry pr. afsnitstype
            for atype, farve in FARVER.items():
                if atype in adt_filtered["VALUE"].values:
                    fig.add_trace(go.Bar(
                        x=[None], y=[None],
                        orientation="h",
                        marker=dict(color=farve),
                        name=atype,
                        showlegend=True,
                    ))

            fig.update_layout(
                barmode="overlay",
                height=80 + len(adt_filtered["VALUE"].unique()) * 60,
                xaxis=dict(title="Timer fra forløbsstart", showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Rå tabel ──────────────────────────────────────────────────────
            with st.expander("Se alle afsnitsregistreringer"):
                tabel = adt_filtered[["TIMESTAMP", "END_TIMESTAMP", "VALUE", "Afsnit"]].copy()
                tabel["TIMESTAMP"] = tabel["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
                tabel["END_TIMESTAMP"] = tabel["END_TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
                tabel.columns = ["Ind", "Ud", "Type", "Afsnit"]
                st.dataframe(tabel.reset_index(drop=True), hide_index=True, use_container_width=True)

    with tab_adt_alle:
        adt_pid2 = get_patient_adt_all(int(selected_pid))
        if adt_pid2.empty:
            st.info("Ingen afsnitsregistreringer for denne patient.")
        else:
            adt_pid2["varighed_timer"] = (
                (adt_pid2["Flyt_ud"] - adt_pid2["Flyt_ind"])
                .dt.total_seconds() / 3600
            )

            # ── Gantt-plot ────────────────────────────────────────────────────
            st.markdown("**Alle afsnit over forløbet**")
            afsnit_liste = adt_pid2["Afsnit"].unique().tolist()
            FARVER = [
                "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"
            ]
            farve_map = {a: FARVER[i % len(FARVER)] for i, a in enumerate(afsnit_liste)}

            fig = go.Figure()
            vist = set()
            for _, r in adt_pid2.iterrows():
                if pd.isna(r["Flyt_ud"]):
                    continue
                varighed = (r["Flyt_ud"] - r["Flyt_ind"]).total_seconds() / 3600
                offset = (r["Flyt_ind"] - patient_start).total_seconds() / 3600
                farve = farve_map[r["Afsnit"]]
                fig.add_trace(go.Bar(
                    x=[varighed],
                    y=[r["Afsnit"]],
                    base=[offset],
                    orientation="h",
                    marker=dict(color=farve, opacity=0.85),
                    name=r["Afsnit"],
                    showlegend=r["Afsnit"] not in vist,
                    hovertemplate=(
                        f"<b>{r['Afsnit']}</b><br>"
                        f"Ind: {r['Flyt_ind'].strftime('%Y-%m-%d %H:%M')}<br>"
                        f"Ud: {r['Flyt_ud'].strftime('%Y-%m-%d %H:%M')}<br>"
                        f"Varighed: {varighed:.1f} timer"
                        "<extra></extra>"
                    ),
                ))
                vist.add(r["Afsnit"])

            fig.update_layout(
                barmode="overlay",
                height=max(300, 80 + len(afsnit_liste) * 40),
                xaxis=dict(title="Timer fra forløbsstart", showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Tabel ─────────────────────────────────────────────────────────
            with st.expander("Se alle afsnitsregistreringer"):
                tabel = adt_pid2[["Flyt_ind", "Flyt_ud", "Afsnit", "varighed_timer"]].copy()
                tabel["Flyt_ind"] = tabel["Flyt_ind"].dt.strftime("%Y-%m-%d %H:%M")
                tabel["Flyt_ud"] = tabel["Flyt_ud"].dt.strftime("%Y-%m-%d %H:%M")
                tabel["varighed_timer"] = tabel["varighed_timer"].round(1)
                tabel.columns = ["Ind", "Ud", "Afsnit", "Timer"]
                st.dataframe(tabel.reset_index(drop=True), hide_index=True, use_container_width=True)

with tab_trauma:
    pid_int = int(selected_pid)

    # ── ISS ──────────────────────────────────────────────────────────────
    st.markdown("### ISS (Injury Severity Score)")
    iss_df = _get_iss_from_notes()
    iss_pid = iss_df[iss_df["PID"] == pid_int]
    if iss_pid.empty:
        st.info("Ingen ISS fundet i notater for denne patient.")
    else:
        iss_row = iss_pid.iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("ISS", int(iss_row["VALUE"]))
        col2.metric("Tidspunkt", pd.to_datetime(iss_row["TIMESTAMP"]).strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    # ── Intubation ───────────────────────────────────────────────────────
    st.markdown("### Intubation")
    intub_df = _get_intubation_from_notes()
    intub_pid = intub_df[intub_df["PID"] == pid_int]
    if intub_pid.empty:
        st.info("Ingen intubation registreret i notater for denne patient.")
    else:
        intub_row = intub_pid.iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("Intuberet", "Ja")
        col2.metric("Tidspunkt", pd.to_datetime(intub_row["TIMESTAMP"]).strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    # ── Cardiac Arrest ───────────────────────────────────────────────────
    st.markdown("### Hjertestop")
    ca_df = _get_cardiac_arrest_from_notes()
    ca_pid = ca_df[ca_df["PID"] == pid_int]
    if ca_pid.empty:
        st.info("Ingen hjertestop registreret i notater for denne patient.")
    else:
        st.markdown(f"**{len(ca_pid)} hjertestop-event(s)**")
        ca_pid = ca_pid.copy()
        ca_pid["TIMESTAMP"] = pd.to_datetime(ca_pid["TIMESTAMP"])
        ca_display = tilfoej_timer(ca_pid, "TIMESTAMP")

        if ca_display.empty:
            st.info(f"Ingen events indenfor de første {tidsvindue} timer.")
        else:
            tabel = ca_display[["TIMESTAMP"]].copy()
            tabel["TIMESTAMP"] = tabel["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M")
            tabel["timer_ind"] = ca_display["timer_siden_start"].round(1)
            tabel.columns = ["Tidspunkt", "Timer fra start"]
            st.dataframe(tabel.reset_index(drop=True), hide_index=True, use_container_width=False,
                         height=min(HEADER_HEIGHT + len(tabel) * ROW_HEIGHT, 300))

            # Timeline plot
            if len(ca_display) >= 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ca_display["timer_siden_start"],
                    y=[1] * len(ca_display),
                    mode="markers",
                    marker=dict(color="#e15759", size=14, symbol="x"),
                    hovertemplate="Timer: %{x:.1f}<extra>Hjertestop</extra>",
                ))
                fig.update_layout(
                    height=120,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(title="Timer fra forløbsstart", showgrid=True, gridcolor="#f0f0f0"),
                    yaxis=dict(visible=False),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

with tab_diag:
    diag_pid = get_patient_diagnoser(int(selected_pid))
    if diag_pid.empty:
        st.info("Ingen diagnoser registreret for denne patient.")
    else:
        diag_pid["dage_siden_start"] = (
            (diag_pid["Noteret_dato"] - patient_start).dt.total_seconds() / 3600 / 24
        )
        # ── Tidslinje ─────────────────────────────────────────────────────
        st.markdown("**Diagnoser over forløbet**")
        fig = go.Figure()
        for i, (_, r) in enumerate(diag_pid.iterrows()):
            fig.add_trace(go.Scatter(
                x=[r["Noteret_dato"]],
                y=[i],
                mode="markers+text",
                marker=dict(size=10, color="#4e79a7"),
                text=[r["Diagnose"]],
                textposition="middle right",
                hovertemplate=f"<b>{r['Diagnose']}</b><br>{r['Noteret_dato'].strftime('%Y-%m-%d')}<extra></extra>",
                showlegend=False,
            ))
        fig.update_layout(
            height=max(200, 60 + len(diag_pid) * 40),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Dato", tickformat="%Y-%m-%d"),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=300, t=20, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        # ── Tabel ─────────────────────────────────────────────────────────
        tabel = diag_pid[["Noteret_dato", "Diagnose"]].copy()
        tabel["Noteret_dato"] = tabel["Noteret_dato"].dt.strftime("%Y-%m-%d")
        tabel.columns = ["Dato", "Diagnose"]
        st.dataframe(
            tabel.reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            height=min(HEADER_HEIGHT + len(tabel) * ROW_HEIGHT, 400),
        )

with tab_overview:
    st.markdown("**Tidsvindue**")
    ov_max = round(forløb_timer, 1)
    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        ov_start = st.number_input("Fra (timer)", min_value=0.0, max_value=ov_max, value=0.0, step=0.5)
    with ov_col2:
        ov_slut = st.number_input("Til (timer)", min_value=0.0, max_value=ov_max, value=ov_max, step=0.5)

    ov_start_dt = patient_start + pd.Timedelta(hours=ov_start)
    ov_slut_dt  = patient_start + pd.Timedelta(hours=ov_slut)

    st.markdown("---")

    col_vit, col_lab = st.columns(2)
    col_icu, col_med = st.columns(2)
    col_proc, col_afsnit = st.columns(2)

    # ── Vitale værdier ────────────────────────────────────────────────────────
    with col_vit:
        st.markdown("**Vitale værdier**")
        vit_ov = get_patient_vitals(int(selected_pid))
        if not vit_ov.empty:
            vit_ov["TIMESTAMP"] = pd.to_datetime(vit_ov["TIMESTAMP"])
            vit_ov = vit_ov[(vit_ov["TIMESTAMP"] >= ov_start_dt) & (vit_ov["TIMESTAMP"] <= ov_slut_dt)]
        if vit_ov.empty:
            st.caption("Ingen målinger i dette vindue.")
        else:
            summary = vit_ov.groupby("FEATURE")["VALUE"].agg(
                Antal="count", Min="min", Max="max", Gns="mean"
            ).round(1).reset_index()
            summary.columns = ["Feature", "Antal", "Min", "Max", "Gns"]
            st.dataframe(summary, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(summary) * ROW_HEIGHT, 300))

    # ── Laboratoriesvar ───────────────────────────────────────────────────────
    with col_lab:
        st.markdown("**Laboratoriesvar**")
        lab_ov = get_patient_labs(int(selected_pid))
        if not lab_ov.empty:
            lab_ov["TIMESTAMP"] = pd.to_datetime(lab_ov["TIMESTAMP"])
            lab_ov = lab_ov[(lab_ov["TIMESTAMP"] >= ov_start_dt) & (lab_ov["TIMESTAMP"] <= ov_slut_dt)]
        if lab_ov.empty:
            st.caption("Ingen målinger i dette vindue.")
        else:
            summary = lab_ov.groupby("FEATURE")["VALUE"].agg(
                Antal="count", Min="min", Max="max", Gns="mean"
            ).round(2).reset_index()
            summary.columns = ["Feature", "Antal", "Min", "Max", "Gns"]
            st.dataframe(summary, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(summary) * ROW_HEIGHT, 300))


    # ── ICU ───────────────────────────────────────────────────────────────────
    with col_icu:
        st.markdown("**ICU**")
        icu_ov = get_patient_icu(int(selected_pid))
        if not icu_ov.empty:
            icu_ov["TIMESTAMP"] = pd.to_datetime(icu_ov["TIMESTAMP"])
            icu_ov = icu_ov[(icu_ov["TIMESTAMP"] >= ov_start_dt) & (icu_ov["TIMESTAMP"] <= ov_slut_dt)]
        if icu_ov.empty:
            st.caption("Ingen ICU-målinger i dette vindue.")
        else:
            summary = icu_ov.groupby("FEATURE")["VALUE"].agg(
                Antal="count", Min="min", Max="max", Gns="mean"
            ).round(1).reset_index()
            summary.columns = ["Feature", "Antal", "Min", "Max", "Gns"]
            st.dataframe(summary, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(summary) * ROW_HEIGHT, 300))

    # ── Medicin ───────────────────────────────────────────────────────────────
    with col_med:
        st.markdown("**Medicin**")
        med_ov = get_patient_medicin(int(selected_pid))
        if not med_ov.empty:
            med_ov["TIMESTAMP"] = pd.to_datetime(med_ov["TIMESTAMP"])
            med_ov = med_ov[(med_ov["TIMESTAMP"] >= ov_start_dt) & (med_ov["TIMESTAMP"] <= ov_slut_dt)]
        if med_ov.empty:
            st.caption("Ingen medicin i dette vindue.")
        else:
            med_tbl = (
                med_ov.groupby(["VALUE", "Generisk_navn"])
                .size()
                .reset_index(name="Antal")
                .sort_values(["VALUE", "Antal"], ascending=[True, False])
            )
            med_tbl.columns = ["Kategori", "Præparat", "Antal adm."]
            st.dataframe(med_tbl, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(med_tbl) * ROW_HEIGHT, 300))
    # ── Procedurer ────────────────────────────────────────────────────────────
    with col_proc:
        st.markdown("**Procedurer**")
        proc_ov = get_patient_procedurer(int(selected_pid))
        if not proc_ov.empty:
            proc_ov["TIMESTAMP"] = pd.to_datetime(proc_ov["TIMESTAMP"])
            proc_ov = proc_ov[(proc_ov["TIMESTAMP"] >= ov_start_dt) & (proc_ov["TIMESTAMP"] <= ov_slut_dt)]
        if proc_ov.empty:
            st.caption("Ingen procedurer i dette vindue.")
        else:
            proc_tbl = (
                proc_ov.groupby("VALUE")["ProcedureName"]
                .apply(lambda x: ", ".join(sorted(x.unique())))
                .reset_index()
            )
            proc_tbl.columns = ["Kategori", "Procedurer"]
            st.dataframe(proc_tbl, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(proc_tbl) * ROW_HEIGHT, 300))

    # ── Afsnit (klassificeret) ────────────────────────────────────────────────
    with col_afsnit:
        st.markdown("**Afsnit**")
        adt_ov = get_patient_adt(int(selected_pid), df)
        if not adt_ov.empty:
            adt_ov = adt_ov[(adt_ov["TIMESTAMP"] >= ov_start_dt) & (adt_ov["TIMESTAMP"] <= ov_slut_dt)]
        if adt_ov.empty:
            st.caption("Ingen afsnitsregistreringer i dette vindue.")
        else:
            adt_ov["varighed_timer"] = (
                (adt_ov["END_TIMESTAMP"] - adt_ov["TIMESTAMP"])
                .dt.total_seconds() / 3600
            ).round(1)
            adt_tbl = adt_ov[["VALUE", "Afsnit", "varighed_timer"]].copy()
            adt_tbl.columns = ["Type", "Afsnit", "Timer"]
            st.dataframe(adt_tbl, hide_index=True, use_container_width=True,
                         height=min(HEADER_HEIGHT + len(adt_tbl) * ROW_HEIGHT, 300))


with tab_notater:
    noter_pid = get_patient_notater(int(selected_pid))

    if noter_pid.empty:
        st.info("Ingen notater for denne patient.")
    else:

        # ── Filtre ────────────────────────────────────────────────────────────
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            dato_min = noter_pid["Oprettelsestidspunkt"].min().date()
            dato_max = noter_pid["Oprettelsestidspunkt"].max().date()
            dato_range = st.date_input("Dato", value=(dato_min, dato_max),
                                       min_value=dato_min, max_value=dato_max)
        with filter_col2:
            notetypes = ["Alle"] + sorted(noter_pid["Notetype"].astype(str).unique().tolist())
            valgt_notetype = st.selectbox("Notetype", notetypes)
        with filter_col3:
            specialer = ["Alle"] + sorted(noter_pid["Speciale"].astype(str).unique().tolist())
            valgt_speciale = st.selectbox("Speciale", specialer)

        fritekst = st.text_input("Søg i noteindhold", placeholder="Skriv søgeord …")

        # Anvend filtre
        filtreret = noter_pid.copy()
        if len(dato_range) == 2:
            filtreret = filtreret[
                (filtreret["Oprettelsestidspunkt"].dt.date >= dato_range[0]) &
                (filtreret["Oprettelsestidspunkt"].dt.date <= dato_range[1])
            ]
        if valgt_notetype != "Alle":
            filtreret = filtreret[filtreret["Notetype"].astype(str) == valgt_notetype]
        if valgt_speciale != "Alle":
            filtreret = filtreret[filtreret["Speciale"].astype(str) == valgt_speciale]
        if fritekst:
            filtreret = filtreret[filtreret["Note"].str.contains(fritekst, case=False, na=False)]

        filtreret = filtreret.reset_index(drop=True)
        st.caption(f"{len(filtreret)} notater")

        # Gem filtreret data i session_state så fragment kan tilgå den
        st.session_state["noter_filtreret"] = filtreret

        @st.fragment
        def vis_notater():
            filtreret = st.session_state.get("noter_filtreret", pd.DataFrame())
            if filtreret.empty:
                st.info("Ingen notater matcher filteret.")
                return

            liste_col, note_col = st.columns([1, 2])

            with liste_col:
                tabel_noter = filtreret[["Oprettelsestidspunkt", "Notetype"]].copy()
                tabel_noter["Oprettelsestidspunkt"] = tabel_noter["Oprettelsestidspunkt"].dt.strftime("%Y-%m-%d %H:%M")
                tabel_noter.columns = ["Tidspunkt", "Type"]

                valgt = st.dataframe(
                    tabel_noter,
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                    on_select="rerun",
                    selection_mode="single-row",
                )
            with note_col:
                valgte_rækker = valgt.selection.rows if valgt.selection.rows else [0]
                idx = valgte_rækker[0]
                r = filtreret.iloc[idx]

                # Beregn tid i forløb
                tid_sekunder = (r["Oprettelsestidspunkt"] - patient_start).total_seconds()
                if tid_sekunder < 0:
                    tid_label = "Før forløbsstart"
                elif tid_sekunder < 86400:
                    t = int(tid_sekunder // 3600)
                    m = int((tid_sekunder % 3600) // 60)
                    tid_label = f"{t}t {m}m inde i forløbet"
                else:
                    d = int(tid_sekunder // 86400)
                    t = int((tid_sekunder % 86400) // 3600)
                    tid_label = f"{d}d {t}t inde i forløbet"

                st.markdown(
                    f"""<div style='font-size:0.78rem; color:#888; margin-bottom:12px; line-height:1.8'>
                        <b>Oprettet:</b> {r['Oprettelsestidspunkt'].strftime('%Y-%m-%d %H:%M')}
                        &nbsp;·&nbsp; {tid_label}<br>
                        <b>Redigeret:</b> {r['Redigeringstidspunkt']}<br>
                        <b>Notetype:</b> {r['Notetype']}
                        &nbsp;·&nbsp; <b>Speciale:</b> {r['Speciale']}
                    </div>""",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""<div style='
                        font-size:0.95rem;
                        line-height:1.7;
                        color:#1a1a1a;
                        background:#fafafa;
                        border-radius:8px;
                        padding:16px 20px;
                        height:450px;
                        overflow-y:auto;
                        white-space:pre-wrap;
                        font-family:Georgia, serif;
                    '>{r["Note"]}</div>""",
                    unsafe_allow_html=True
                )

        vis_notater()
