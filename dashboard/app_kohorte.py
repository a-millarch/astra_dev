"""
ASTRA Kohorte Dashboard - Redesigned
Clean, structured cohort-level analytics

Run: streamlit run dashboard/kohorte_dashboard.py
"""

import sys
import os

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from astra.utils import get_base_df, get_cfg
from astra.data.filters import (
    filter_vitals, filter_labs, filter_ita, 
    filter_medicin, filter_procedures, filter_adt
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ASTRA Kohorte",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM STYLING - Clean, minimal design
# =============================================================================
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {background-color: #fafafa;}
    section[data-testid="stSidebar"] .stRadio > label {font-weight: 600; color: #333;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: 2px solid #e0e0e0;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: transparent;
        border: none;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background: #f0f4f8;
        color: #1a3a6b;
        border-radius: 4px 4px 0 0;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {font-size: 1.8rem; color: #1a3a6b;}
    [data-testid="stMetricLabel"] {font-size: 0.85rem; color: #666; text-transform: uppercase;}
    
    /* Headers */
    h1 {color: #1a3a6b; font-weight: 700; margin-bottom: 0.5rem;}
    h2 {color: #333; font-weight: 600; font-size: 1.3rem; margin-top: 1.5rem;}
    h3 {color: #444; font-weight: 600; font-size: 1.1rem;}
    
    /* Expander */
    .streamlit-expanderHeader {font-weight: 600; color: #1a3a6b;}
    
    /* Dataframes */
    .stDataFrame {border-radius: 4px;}
    
    /* Filter badge */
    .filter-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a3a6b;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADERS (CACHED)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_base():
    base = get_base_df()
    for col in ["start", "end", "DOB", "DOD"]:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], errors="coerce")
    return base


@st.cache_resource(show_spinner=False)
def load_holdout_pids():
    """Load holdout PIDs from cached file."""
    # Try multiple possible locations for holdout PIDs
    possible_paths = [
        os.path.join(REPO_ROOT, "data/processed/holdout_pids.csv"),
        os.path.join(REPO_ROOT, "data/interim/holdout_pids.csv"),
        os.path.join(REPO_ROOT, "models/holdout_pids.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                pid_col = next((c for c in df.columns if c.upper() == "PID"), df.columns[0])
                return set(df[pid_col].dropna().unique())
            except Exception:
                continue
    
    # Try pickle format
    pickle_paths = [
        os.path.join(REPO_ROOT, "data/processed/holdout_pids.pkl"),
        os.path.join(REPO_ROOT, "data/interim/holdout_pids.pkl"),
    ]
    
    for path in pickle_paths:
        if os.path.exists(path):
            try:
                pids = pd.read_pickle(path)
                if isinstance(pids, pd.DataFrame):
                    pid_col = next((c for c in pids.columns if c.upper() == "PID"), pids.columns[0])
                    return set(pids[pid_col].dropna().unique())
                elif isinstance(pids, (list, set, np.ndarray, pd.Series)):
                    return set(pids)
            except Exception:
                continue

    # Auto-generate from data pipeline as last resort
    try:
        from astra.data.caching import prepare_data_and_dls_cached
        cfg = get_cfg()
        data = prepare_data_and_dls_cached(cfg)
        holdout_pids = data["holdout"].base["PID"].unique()
        # Save for future use
        save_path = os.path.join(REPO_ROOT, "data/interim/holdout_pids.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame({"PID": holdout_pids}).to_csv(save_path, index=False)
        return set(holdout_pids)
    except Exception:
        pass

    return None


@st.cache_resource(show_spinner=False)
def load_vitals():
    return pd.read_pickle("data/interim/concepts/VitaleVaerdier.pkl")

@st.cache_resource(show_spinner=False)
def load_labs():
    return pd.read_pickle("data/interim/concepts/Labsvar.pkl")

@st.cache_resource(show_spinner=False)
def load_icu():
    return pd.read_pickle("data/interim/concepts/ITAOversigtsrapport.pkl")

@st.cache_resource(show_spinner=False)
def load_medicin():
    return pd.read_pickle("data/interim/concepts/Medicin.pkl")

@st.cache_resource(show_spinner=False)
def load_procedurer():
    return pd.read_pickle("data/interim/concepts/Procedurer.pkl")

@st.cache_resource(show_spinner=False)
def load_adt():
    return pd.read_pickle("data/interim/concepts/ADTHaendelser.pkl")

@st.cache_resource(show_spinner=False)
def load_diagnoser():
    return pd.read_pickle("data/interim/concepts/Diagnoser.pkl")


# =============================================================================
# COHORT FLAGS & FILTERING
# =============================================================================
@st.cache_resource(show_spinner=False)
def compute_cohort_flags(base):
    """Add Gravid/Psyk flags to base dataframe."""
    proc = load_procedurer()
    gravid_mask = proc["ProcedureName"].astype(str).str.contains(
        r"Gravid|Svangerskab", case=False, na=False
    )
    gravide_pids = set(proc.loc[gravid_mask, "PID"].dropna().unique())
    
    adt = load_adt()
    afsnit_col = next((c for c in ["Afsnit", "afsnit"] if c in adt.columns), None)
    if afsnit_col:
        psyk_mask = adt[afsnit_col].astype(str).str.contains(
            r"psyk|lukket", case=False, na=False
        )
        psyk_pids = set(adt.loc[psyk_mask, "PID"].dropna().unique())
    else:
        psyk_pids = set()
    
    base = base.copy()
    base["is_gravid"] = base["PID"].isin(gravide_pids).astype(int)
    base["is_psyk"] = base["PID"].isin(psyk_pids).astype(int)
    return base


def apply_filters(base, data_split, exclude_option, subgroup):
    """Apply all cohort filters."""
    df = base.copy()
    
    # Data split filter
    if data_split == "Holdout":
        holdout_pids = load_holdout_pids()
        if holdout_pids is not None:
            df = df[df["PID"].isin(holdout_pids)]
        elif "split" in df.columns:
            df = df[df["split"] == "holdout"]
        elif "is_holdout" in df.columns:
            df = df[df["is_holdout"] == 1]
        else:
            st.warning("Holdout PIDs ikke fundet. Gem holdout_pids.csv i data/processed/")
    
    # Exclusion filters
    if exclude_option == "Gravide":
        if "is_gravid" in df.columns:
            df = df[df["is_gravid"] != 1]
    elif exclude_option == "Psykiatri":
        if "is_psyk" in df.columns:
            df = df[df["is_psyk"] != 1]
    elif exclude_option == "Gravide & Psykiatri":
        if "is_gravid" in df.columns and "is_psyk" in df.columns:
            df = df[(df["is_gravid"] != 1) & (df["is_psyk"] != 1)]
    
    # Subgroup filters
    if subgroup == "LOS > 2d + død":
        if "start" in df.columns and "end" in df.columns:
            df["_los"] = (df["end"] - df["start"]).dt.total_seconds() / 86400
            mort_col = next((c for c in ["deceased_30d", "deceased_90d"] if c in df.columns), None)
            if mort_col:
                df = df[(df["_los"] > 2) & (df[mort_col] > 0)]
            df = df.drop(columns=["_los"], errors="ignore")
    
    elif subgroup == "LOS > 4d + overlevet":
        if "start" in df.columns and "end" in df.columns:
            df["_los"] = (df["end"] - df["start"]).dt.total_seconds() / 86400
            mort_col = next((c for c in ["deceased_30d", "deceased_90d"] if c in df.columns), None)
            if mort_col:
                df = df[(df["_los"] > 4) & (df[mort_col] == 0)]
            df = df.drop(columns=["_los"], errors="ignore")
    
    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def filter_to_cohort(df, pid_set):
    """Filter dataframe to cohort PIDs."""
    return df[df["PID"].isin(pid_set)].copy()


def add_hours_from_start(df, ts_col, base):
    """Add hours_from_start column."""
    starts = base[["PID", "start"]].drop_duplicates("PID")
    df = df.merge(starts, on="PID", how="left", suffixes=("", "_base"))
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["hours_from_start"] = (df[ts_col] - df["start"]).dt.total_seconds() / 3600
    return df


def format_number(n):
    """Format large numbers with thousand separators."""
    return f"{n:,.0f}".replace(",", " ")


# =============================================================================
# SIDEBAR - Cohort Selection
# =============================================================================
with st.sidebar:
    st.markdown("## Kohorte")
    
    # Data split
    data_split = st.radio(
        "Datasplit",
        ["Alle", "Holdout"],
        index=0,
        help="Vælg hvilken del af data der skal vises"
    )
    
    st.markdown("---")
    st.markdown("### Ekskluderinger")
    
    exclude_option = st.radio(
        "Ekskluder",
        ["Ingen", "Gravide", "Psykiatri", "Gravide & Psykiatri"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Subgruppe")
    
    subgroup = st.radio(
        "Vælg subgruppe",
        ["Alle", "LOS > 2d + død", "LOS > 4d + overlevet"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Load and filter data
    with st.spinner("Indlæser data..."):
        base_raw = load_base()
        base_flagged = compute_cohort_flags(base_raw)
        base = apply_filters(base_flagged, data_split, exclude_option, subgroup)
    
    # Show cohort stats in sidebar
    n_patients = base["PID"].nunique()
    st.markdown(f"**Forløb:** {format_number(n_patients)}")
    
    if "CPR_hash" in base.columns:
        n_unique = base["CPR_hash"].nunique()
        st.markdown(f"**Unikke patienter:** {format_number(n_unique)}")


# =============================================================================
# MAIN CONTENT - Header with active filters
# =============================================================================
st.markdown("# Kohorte Oversigt")

# Show active filters as badges
active_filters = []
if data_split != "Alle":
    active_filters.append(data_split)
if exclude_option != "Ingen":
    active_filters.append(f"Uden {exclude_option.lower()}")
if subgroup != "Alle":
    active_filters.append(subgroup)

if active_filters:
    badges = " ".join([f'<span class="filter-badge">{f}</span>' for f in active_filters])
    st.markdown(f"{badges}", unsafe_allow_html=True)

st.markdown("---")


# =============================================================================
# TABS - Main navigation
# =============================================================================
tabs = st.tabs([
    "Overblik",
    "Kirurgi",
    "Vitale værdier", 
    "Laboratoriesvar",
    "ICU",
    "Medicin",
    "Procedurer",
    "Afsnit",
    "Diagnoser"
])


# =============================================================================
# TAB: OVERBLIK
# =============================================================================
with tabs[0]:
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    base["age_years"] = (base["start"] - base["DOB"]).dt.days / 365.25
    completed = base[base["end"].notna()].copy()
    
    with col1:
        st.metric("Forløb", format_number(base["PID"].nunique()))
    
    with col2:
        if "CPR_hash" in base.columns:
            st.metric("Unikke patienter", format_number(base["CPR_hash"].nunique()))
        else:
            st.metric("Unikke patienter", "N/A")
    
    with col3:
        if not completed.empty:
            inhosp = completed[
                completed["DOD"].notna() &
                (completed["DOD"] >= completed["start"]) &
                (completed["DOD"] <= completed["end"])
            ]
            pct = (inhosp["PID"].nunique() / completed["PID"].nunique() * 100)
            st.metric("In-hospital død", f"{pct:.1f}%")
        else:
            st.metric("In-hospital doød", "N/A")
    
    with col4:
        mort_col = next((c for c in ["deceased_30d", "deceased_90d"] if c in base.columns), None)
        if mort_col:
            pct = (base[mort_col].sum() / len(base) * 100)
            st.metric(f"{mort_col.replace('_', ' ').title()}", f"{pct:.1f}%")
        else:
            st.metric("Mortalitet", "N/A")
    
    st.markdown("---")
    
    # Demographics row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Kønsfordeling")
        sex_counts = base["SEX"].value_counts().reset_index()
        sex_counts.columns = ["Køn", "Antal"]
        fig_sex = px.pie(
            sex_counts, values="Antal", names="Køn",
            color_discrete_sequence=["#4e79a7", "#f28e2b"],
            hole=0.4
        )
        fig_sex.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=250,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    
    with col_right:
        st.markdown("### Aldersfordeling")
        mean_age = base["age_years"].mean()
        median_age = base["age_years"].median()
        st.markdown(f'<span style="color: #e15759; font-weight: 600;">Gennemsnit: {mean_age:.1f} aar</span> | Median: {median_age:.1f} år', unsafe_allow_html=True)
        
        fig_age = px.histogram(
            base.dropna(subset=["age_years"]),
            x="age_years",
            nbins=25,
            color_discrete_sequence=["#4e79a7"]
        )
        fig_age.add_vline(x=mean_age, line_dash="dash", line_color="#e15759", 
                         annotation_text=f"Gns: {mean_age:.1f}",
                         annotation_font_color="#e15759")
        fig_age.update_layout(
            margin=dict(l=20, r=20, t=20, b=40),
            height=250,
            xaxis_title="Alder (aar)",
            yaxis_title="Antal",
            bargap=0.1
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    st.markdown("---")
    
    # Visitation type
    st.markdown("### Visitationstype")
    
    if "type_visitation" in base.columns:
        tv = base["type_visitation"].fillna("Ukendt").str.strip().str.casefold()
        total = base["PID"].nunique()
        
        primaer = base[tv == "primær"]["PID"].nunique()
        sekundaer = base[tv == "sekundær"]["PID"].nunique()
        primaer_ingen = base[tv == "primær ingen rh"]["PID"].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Primær", primaer, f"{primaer/total*100:.1f}%" if total else None)
        with col2:
            st.metric("Sekundær", sekundaer, f"{sekundaer/total*100:.1f}%" if total else None)
        with col3:
            st.metric("Primær uden RH", primaer_ingen, f"{primaer_ingen/total*100:.1f}%" if total else None)
        with col4:
            other = total - primaer - sekundaer - primaer_ingen
            st.metric("Andet/ukendt", other)
    else:
        st.info("Visitationstype ikke tilgængelig i data.")
    
    st.markdown("---")
    
    # Secondary transport to RH
    st.markdown("### Sekundær transport til RH")
    
    if "type_visitation" in base.columns and "time_to_RH" in base.columns:
        tv = base["type_visitation"].fillna("").str.strip().str.casefold()
        sec = base[tv == "sekundær"].copy()
        sec["time_to_RH"] = pd.to_timedelta(sec["time_to_RH"], errors="coerce")
        sec["ttrh_hours"] = sec["time_to_RH"].dt.total_seconds() / 3600
        sec = sec.dropna(subset=["ttrh_hours"])
        sec = sec[sec["ttrh_hours"] >= 0]
        
        if sec.empty:
            st.info("Ingen sekundære forløb med gyldig tid til RH.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sekundære forløb", sec["PID"].nunique())
            with col2:
                st.metric("Median tid til RH", f"{sec['ttrh_hours'].median():.1f} t")
            with col3:
                st.metric("p25 / p75", f"{sec['ttrh_hours'].quantile(0.25):.1f} / {sec['ttrh_hours'].quantile(0.75):.1f} t")
    else:
        st.info("Transport-data ikke tilgængelig.")
    
    st.markdown("---")
    
    # LOS distribution
    st.markdown("### Forløbsvarighed (LOS)")
    
    los = base.dropna(subset=["start", "end"]).copy()
    los["los_hours"] = (los["end"] - los["start"]).dt.total_seconds() / 3600
    los = los[los["los_hours"] >= 0]
    
    if not los.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median", f"{los['los_hours'].median():.1f} t")
        with col2:
            st.metric("Gennemsnit", f"{los['los_hours'].mean():.1f} t")
        with col3:
            st.metric("p25 / p75", f"{los['los_hours'].quantile(0.25):.0f} / {los['los_hours'].quantile(0.75):.0f} t")
        with col4:
            st.metric("Max", f"{los['los_hours'].max():.0f} t")
        
        # Split histogram for short vs long stays
        CUT = 168  # 7 days
        los_short = los[los["los_hours"] <= CUT]
        
        fig_los = px.histogram(
            los_short, x="los_hours", nbins=40,
            color_discrete_sequence=["#4e79a7"]
        )
        fig_los.update_layout(
            margin=dict(l=20, r=20, t=20, b=40),
            height=220,
            xaxis_title="Timer",
            yaxis_title="Antal",
            bargap=0.1
        )
        st.plotly_chart(fig_los, use_container_width=True)
        
        st.caption(f"Viser forløb <= {CUT} timer. {len(los[los['los_hours'] > CUT])} forløb er længere.")


# =============================================================================
# TAB: KIRURGI
# =============================================================================
with tabs[1]:
    st.markdown("### Kirurgi")
    
    pid_set = set(base["PID"].dropna().unique())
    proc_raw = load_procedurer()
    
    # Filter to K-codes (surgical procedures)
    op = proc_raw[proc_raw["ProcedureCode"].astype(str).str.startswith("K")].copy()
    op = filter_to_cohort(op, pid_set)
    
    if op.empty:
        st.info("Ingen kirurgiske procedurer for denne kohorte.")
    else:
        # Merge with base for demographics
        op = op.merge(base[["PID", "DOB", "SEX", "start"]], on="PID", how="left")
        for col in ["DOB", "start", "ServiceDatetime"]:
            if col in op.columns:
                op[col] = pd.to_datetime(op[col], errors="coerce")
        op["age_years"] = (op["ServiceDatetime"] - op["DOB"]).dt.days / 365.25
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        n_patients_op = op["PID"].nunique()
        ops_per_pid = op.groupby("PID").size()
        
        # Time to surgery
        first_op = op.dropna(subset=["ServiceDatetime"]).groupby("PID")["ServiceDatetime"].min().reset_index()
        first_op.columns = ["PID", "first_op"]
        t2s = base[["PID", "start"]].merge(first_op, on="PID", how="inner")
        t2s["t2s_hours"] = (t2s["first_op"] - t2s["start"]).dt.total_seconds() / 3600
        t2s = t2s[t2s["t2s_hours"] >= 0]
        
        with col1:
            st.metric("Forløb med kirurgi", n_patients_op)
        with col2:
            st.metric("Andel af kohorte", f"{n_patients_op / len(pid_set) * 100:.1f}%")
        with col3:
            st.metric("Gns. OP pr. forløb", f"{ops_per_pid.mean():.2f}")
        with col4:
            if not t2s.empty:
                st.metric("Median tid til OP", f"{t2s['t2s_hours'].median():.1f} t")
            else:
                st.metric("Median tid til OP", "N/A")
        
        st.markdown("---")
        
        # Top procedures
        st.markdown("#### Hyppigste operationer")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("Antal", 5, 30, 10, key="kir_topn")
        
        top_ops = op.groupby(["ProcedureCode", "ProcedureName"]).agg(
            Antal=("PID", "count"),
            Patienter=("PID", "nunique")
        ).reset_index().sort_values("Patienter", ascending=False).head(top_n)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(top_ops, use_container_width=True, hide_index=True)
        with col2:
            top_ops["label"] = top_ops["ProcedureCode"] + " - " + top_ops["ProcedureName"].str[:30]
            fig = px.bar(
                top_ops.sort_values("Patienter"),
                x="Patienter", y="label", orientation="h",
                color_discrete_sequence=["#4e79a7"]
            )
            fig.update_layout(
                height=max(250, top_n * 25),
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(categoryorder="total ascending"),
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Body region mapping
        st.markdown("#### Opereret kropsregion")
        
        feature_map = {
            "Neuro": ("KAAA27","KAAD05","KAAF00A","KAAD00","KAAD15","KAAA20","KAAA40","KAAC00","KAAA99","KAAD40","KAAL11","KAAB30","KAAD10","KABC60","KAAD30","KAWD00","KAAK35","KAAK00","KAAK10"),
            "Abdominal": ("KNHJ63","KJBA00","KPCT20","KPCT99","KJDH70","KJJA96","KKBV02A","KJJW96","KKAH00","KJKB30","KKAD10","KKAC00","KPCT30","KJJA50","KJJB00"),
            "Vascular": ("KFNG05A","KFNG02A","KPBH20","KPET11","KPEA12","KPBC30","KPHC23","KPDC30","KPBB30","KPDG10","KPDT30","KPEH12","KPBC10","KPBN20","KACB22","KPAC20","KPBE30","KPDF10","KPEA10","KPBA20","KPHH99","KFCA70","KFCA50","KPBU82","KPHP30","KPEN11","KPEH20","KPFN30","KPEC12","KNDL41","KPDQ10","KPAP21","KPCH30","KPFC10","KPHC22","KPAQ21","KPBC20","KPEP11","KPEU87","KPFE10"),
            "Thorax": ("KGAB10","KGAA31","KGAB20","KGDB11","KGAC10","KFLC00","KFXE00","KFEB10","KFXD00","KFWW96","KGDA40","KGAE30","KUGC02","KFJB00","KGAE03","KGDB10","KGDA41","KFEW96","KGDB96","KGAE96"),
            "Orto": ("KNGJ22","KNAG73","KNAG40","KNFJ54","KNAG70","KNGM09","KNEJ29","KNGJ29","KNGJ52","KNFJ25","KNDL40A","KNEJ69","KACB23","KNGJ21","KNCJ45","KNCJ27","KACB29","KNAG71","KNDA02","KACC51","KNHJ45","KNFJ51","KNAG72","KNDM09","KNHJ62","KNDJ42","KNFJ43","KNBQ03","KNCJ65","KNGQ19","KNAG76","KNGJ40","KABC56","KPBB99","KACB21","KNGJ61","KNDL40","KNFQ19","KNAN00","KNBJ41","KNBJ61","KNCJ88","KNBA02","KNHJ80","KNDJ43","KNHJ47","KNGE29","KNHJ23","KNHJ71","KACA13","KNFJ10","KNFJ70","KNFJ73","KNHN09","KNCJ67","KNGJ71","KNCJ26","KNCJ60","KNCJ42","KNAN03","KNFJ52","KNCE22","KNDQ99","KNHQ22","KNCL49","KQCG30","KNCJ64","KNAN02","KNAK12","KNHJ72","KABA00","KNCJ28","KNCJ80","KNFJ44","KNHJ82","KNFJ55","KNEJ89","KNAJ12","KACC29","KNDJ11","KNDU39","KNDJ70","KNBJ51","KNHJ22","KNHL49","KNHE99","KNFM09","KNGJ80","KQAA10","KNHJ14","KNHJ44","KNDL41A","KNAK10","KNBJ62","KNBJ21","KNCJ47","KNAJ00","KACA19","KNFQ99","KNFJ50","KNGJ73","KNHJ81","KNGM99","KECB40","KNGD22","KNCJ05","KNHJ25","KACC53","KNHJ24","KNCM09","KNDH12","KNAN04","KNFJ65","KNDH02","KNHJ41","KNHJ74","KNCJ66","KNGJ63","KNHJ42","KNFJ45","KNGJ42","KNAG41","KNFA02A"),
            "ENT": ("KEFB20","KEDC38","KEEC25","KEEC35","KDLD30","KEWE00","KECB20A","KDQE00","KEDC36","KGBA00","KGAB00","KDWE00","KENC00","KDHD30","KDJD20","KDAD30","KDWA00","KDQW99","KEMC00","KEDC39B","KDLD20"),
        }
        
        code_to_cat = {code: cat for cat, codes in feature_map.items() for code in codes}
        op["region"] = op["ProcedureCode"].astype(str).map(code_to_cat).fillna("Andet")
        
        region_counts = op.groupby("region")["PID"].nunique().reset_index()
        region_counts.columns = ["Region", "Patienter"]
        region_counts = region_counts.sort_values("Patienter", ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(region_counts, use_container_width=True, hide_index=True)
        with col2:
            fig = px.bar(
                region_counts.sort_values("Patienter"),
                x="Patienter", y="Region", orientation="h",
                color_discrete_sequence=["#4e79a7"]
            )
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(categoryorder="total ascending")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Reoperations
        st.markdown("#### Genoperationer")
        st.caption("Samme procedure udført mere end en gang i samme forløb")
        
        reop = op.groupby(["PID", "ProcedureCode", "ProcedureName"]).size().reset_index(name="count")
        reop = reop[reop["count"] >= 2]
        n_reop_patients = reop["PID"].nunique()
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Forløb med genop", n_reop_patients)
        with col2:
            st.metric("Andel af kirurgi", f"{n_reop_patients / n_patients_op * 100:.1f}%")
        
        # Top reoperations
        col1, col2 = st.columns([1, 3])
        with col1:
            top_reop_n = st.slider("Top genoperationer", 5, 30, 10, key="reop_topn")
        
        reop_counts = reop.groupby(["ProcedureCode", "ProcedureName"])["PID"].nunique().reset_index(name="Patienter")
        reop_counts = reop_counts.sort_values("Patienter", ascending=False).head(top_reop_n)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(reop_counts, use_container_width=True, hide_index=True)
        with col2:
            if not reop_counts.empty:
                reop_counts["label"] = reop_counts["ProcedureCode"] + " - " + reop_counts["ProcedureName"].str[:25]
                fig = px.bar(
                    reop_counts.sort_values("Patienter"),
                    x="Patienter", y="label", orientation="h",
                    color_discrete_sequence=["#e15759"]
                )
                fig.update_layout(
                    height=max(200, top_reop_n * 22),
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending"),
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Time profile for surgery categories
        st.markdown("#### Tidsprofil for operationskategorier")
        
        op = add_hours_from_start(op, "ServiceDatetime", base)
        op_valid = op[(op["hours_from_start"] >= 0) & (op["hours_from_start"] <= 168)]
        op_valid["time_bin"] = (op_valid["hours_from_start"] // 4) * 4
        
        ts_op = op_valid.groupby(["time_bin", "region"])["PID"].nunique().reset_index(name="Patienter")
        
        fig_ts = px.line(
            ts_op, x="time_bin", y="Patienter", color="region",
            markers=True,
            color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
        )
        fig_ts.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Timer fra start",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.markdown("---")
        
        # Demographics expander
        with st.expander("Demografi for opererede patienter"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Kønsfordeling**")
                sex_op = op.drop_duplicates("PID")["SEX"].value_counts().reset_index()
                sex_op.columns = ["Køn", "Antal"]
                fig_sex = px.pie(
                    sex_op, values="Antal", names="Køn",
                    color_discrete_sequence=["#4e79a7", "#f28e2b"],
                    hole=0.4
                )
                fig_sex.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_sex, use_container_width=True)
            
            with col2:
                st.markdown("**Aldersfordeling**")
                op_unique = op.drop_duplicates("PID")
                mean_age_op = op_unique["age_years"].mean()
                median_age_op = op_unique["age_years"].median()
                st.markdown(f'<span style="color: #e15759; font-weight: 600;">Gennemsnit: {mean_age_op:.1f} aar</span> | Median: {median_age_op:.1f} aar', unsafe_allow_html=True)
                
                fig_age = px.histogram(
                    op_unique.dropna(subset=["age_years"]),
                    x="age_years", nbins=20,
                    color_discrete_sequence=["#4e79a7"]
                )
                fig_age.add_vline(x=mean_age_op, line_dash="dash", line_color="#e15759",
                                 annotation_font_color="#e15759")
                fig_age.update_layout(
                    height=250, margin=dict(l=20, r=20, t=20, b=40),
                    xaxis_title="Alder (år)", yaxis_title="Antal"
                )
                st.plotly_chart(fig_age, use_container_width=True)


# =============================================================================
# TAB: VITALE VÆRDIER
# =============================================================================
with tabs[2]:
    st.markdown("### Vitale værdier")
    
    pid_set = set(base["PID"].dropna().unique())
    vitals_raw = load_vitals()
    vitals = filter_to_cohort(vitals_raw, pid_set)
    
    if vitals.empty:
        st.info("Ingen vitale værdier for denne kohorte.")
    else:
        vitals = filter_vitals(vitals)
        vitals["VALUE"] = pd.to_numeric(vitals["VALUE"], errors="coerce")
        vitals = vitals.dropna(subset=["VALUE"])
        vitals = add_hours_from_start(vitals, "TIMESTAMP", base)
        
        # Time window filter
        col1, col2 = st.columns([1, 3])
        with col1:
            hours_window = st.slider(
                "Tidsvindue (timer)", 
                min_value=1, max_value=168, value=24,
                key="vitals_hours"
            )
        
        vitals_w = vitals[
            (vitals["hours_from_start"] >= 0) & 
            (vitals["hours_from_start"] <= hours_window)
        ]
        
        if vitals_w.empty:
            st.info("Ingen data i dette tidsvindue.")
        else:
            # Summary stats
            summary = vitals_w.groupby("FEATURE").agg(
                Maalinger=("VALUE", "count"),
                Patienter=("PID", "nunique"),
                Median=("VALUE", "median"),
                p25=("VALUE", lambda x: x.quantile(0.25)),
                p75=("VALUE", lambda x: x.quantile(0.75))
            ).round(1).reset_index()
            summary = summary.sort_values("Patienter", ascending=False)
            
            # Search
            search = st.text_input("Søg feature", placeholder="Fx Puls, BT...", key="vit_search")
            if search:
                summary = summary[summary["FEATURE"].str.contains(search, case=False, na=False)]
            
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                height=min(300, 35 + len(summary) * 35)
            )
            
            # Detail view for selected feature
            st.markdown("---")
            features = summary["FEATURE"].tolist()
            if features:
                selected = st.selectbox("Vælg feature til detaljeret visning", features, key="vit_detail")
                
                feat_data = vitals_w[vitals_w["FEATURE"] == selected]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Tidsprofil (median med IQR)**")
                    feat_data["time_bin"] = (feat_data["hours_from_start"] // 2) * 2
                    ts_agg = feat_data.groupby("time_bin")["VALUE"].agg(
                        median="median",
                        p25=lambda x: x.quantile(0.25),
                        p75=lambda x: x.quantile(0.75)
                    ).reset_index()
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=pd.concat([ts_agg["time_bin"], ts_agg["time_bin"].iloc[::-1]]),
                        y=pd.concat([ts_agg["p75"], ts_agg["p25"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(78,121,167,0.2)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR"
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts_agg["time_bin"],
                        y=ts_agg["median"],
                        mode="lines+markers",
                        line=dict(color="#4e79a7", width=2),
                        name="Median"
                    ))
                    fig_ts.update_layout(
                        height=280,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Timer",
                        showlegend=False
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col2:
                    st.markdown("**Værdifordeling (histogram)**")
                    fig_hist = px.histogram(
                        feat_data, x="VALUE", nbins=30,
                        color_discrete_sequence=["#4e79a7"]
                    )
                    fig_hist.add_vline(x=feat_data["VALUE"].median(), line_dash="dash", line_color="#e15759")
                    fig_hist.update_layout(
                        height=280,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title=selected,
                        yaxis_title="Antal"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB: LABORATORIESVAR
# =============================================================================
with tabs[3]:
    st.markdown("### Laboratoriesvar")
    
    pid_set = set(base["PID"].dropna().unique())
    labs_raw = load_labs()
    labs = filter_to_cohort(labs_raw, pid_set)
    
    if labs.empty:
        st.info("Ingen laboratoriesvar for denne kohorte.")
    else:
        labs = filter_labs(labs)
        labs["VALUE"] = pd.to_numeric(labs["VALUE"], errors="coerce")
        labs = labs.dropna(subset=["VALUE"])
        labs = add_hours_from_start(labs, "TIMESTAMP", base)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            hours_window = st.slider(
                "Tidsvindue (timer)", 
                min_value=1, max_value=168, value=24,
                key="labs_hours"
            )
        
        labs_w = labs[
            (labs["hours_from_start"] >= 0) & 
            (labs["hours_from_start"] <= hours_window)
        ]
        
        if labs_w.empty:
            st.info("Ingen data i dette tidsvindue.")
        else:
            summary = labs_w.groupby("FEATURE").agg(
                Maalinger=("VALUE", "count"),
                Patienter=("PID", "nunique"),
                Median=("VALUE", "median"),
                p25=("VALUE", lambda x: x.quantile(0.25)),
                p75=("VALUE", lambda x: x.quantile(0.75))
            ).round(2).reset_index()
            summary = summary.sort_values("Patienter", ascending=False)
            
            search = st.text_input("Søg feature", placeholder="Fx Hgb, Kreatinin...", key="lab_search")
            if search:
                summary = summary[summary["FEATURE"].str.contains(search, case=False, na=False)]
            
            st.dataframe(summary, use_container_width=True, hide_index=True, height=min(300, 35 + len(summary) * 35))
            
            # Detail view
            st.markdown("---")
            features = summary["FEATURE"].tolist()
            if features:
                selected = st.selectbox("Vælg feature til detaljeret visning", features, key="lab_detail")
                
                feat_data = labs_w[labs_w["FEATURE"] == selected]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Tidsprofil (median med IQR)**")
                    feat_data["time_bin"] = (feat_data["hours_from_start"] // 2) * 2
                    ts_agg = feat_data.groupby("time_bin")["VALUE"].agg(
                        median="median",
                        p25=lambda x: x.quantile(0.25),
                        p75=lambda x: x.quantile(0.75)
                    ).reset_index()
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=pd.concat([ts_agg["time_bin"], ts_agg["time_bin"].iloc[::-1]]),
                        y=pd.concat([ts_agg["p75"], ts_agg["p25"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(78,121,167,0.2)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR"
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts_agg["time_bin"],
                        y=ts_agg["median"],
                        mode="lines+markers",
                        line=dict(color="#4e79a7", width=2),
                        name="Median"
                    ))
                    fig_ts.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Timer", showlegend=False)
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col2:
                    st.markdown("**Værdifordeling (histogram)**")
                    fig_hist = px.histogram(feat_data, x="VALUE", nbins=30, color_discrete_sequence=["#4e79a7"])
                    fig_hist.add_vline(x=feat_data["VALUE"].median(), line_dash="dash", line_color="#e15759")
                    fig_hist.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title=selected, yaxis_title="Antal")
                    st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB: ICU
# =============================================================================
with tabs[4]:
    st.markdown("### ICU-målinger")
    
    pid_set = set(base["PID"].dropna().unique())
    icu_raw = load_icu()
    icu = filter_to_cohort(icu_raw, pid_set)
    
    if icu.empty:
        st.info("Ingen ICU-data for denne kohorte.")
    else:
        icu = filter_ita(icu)
        icu["VALUE"] = pd.to_numeric(icu["VALUE"], errors="coerce")
        icu = icu.dropna(subset=["VALUE"])
        icu = add_hours_from_start(icu, "TIMESTAMP", base)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            hours_window = st.slider(
                "Tidsvindue (timer)", 
                min_value=1, max_value=168, value=24,
                key="icu_hours"
            )
        
        icu_w = icu[
            (icu["hours_from_start"] >= 0) & 
            (icu["hours_from_start"] <= hours_window)
        ]
        
        if icu_w.empty:
            st.info("Ingen data i dette tidsvindue.")
        else:
            summary = icu_w.groupby("FEATURE").agg(
                Maalinger=("VALUE", "count"),
                Patienter=("PID", "nunique"),
                Median=("VALUE", "median"),
                p25=("VALUE", lambda x: x.quantile(0.25)),
                p75=("VALUE", lambda x: x.quantile(0.75))
            ).round(1).reset_index()
            summary = summary.sort_values("Patienter", ascending=False)
            
            search = st.text_input("Søg feature", key="icu_search")
            if search:
                summary = summary[summary["FEATURE"].str.contains(search, case=False, na=False)]
            
            st.dataframe(summary, use_container_width=True, hide_index=True, height=min(300, 35 + len(summary) * 35))
            
            # Detail view
            st.markdown("---")
            features = summary["FEATURE"].tolist()
            if features:
                selected = st.selectbox("Vælg feature til detaljeret visning", features, key="icu_detail")
                
                feat_data = icu_w[icu_w["FEATURE"] == selected]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Tidsprofil (median med IQR)**")
                    feat_data["time_bin"] = (feat_data["hours_from_start"] // 2) * 2
                    ts_agg = feat_data.groupby("time_bin")["VALUE"].agg(
                        median="median",
                        p25=lambda x: x.quantile(0.25),
                        p75=lambda x: x.quantile(0.75)
                    ).reset_index()
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=pd.concat([ts_agg["time_bin"], ts_agg["time_bin"].iloc[::-1]]),
                        y=pd.concat([ts_agg["p75"], ts_agg["p25"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(78,121,167,0.2)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR"
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts_agg["time_bin"],
                        y=ts_agg["median"],
                        mode="lines+markers",
                        line=dict(color="#4e79a7", width=2),
                        name="Median"
                    ))
                    fig_ts.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Timer", showlegend=False)
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col2:
                    st.markdown("**Værdifordeling (histogram)**")
                    fig_hist = px.histogram(feat_data, x="VALUE", nbins=30, color_discrete_sequence=["#4e79a7"])
                    fig_hist.add_vline(x=feat_data["VALUE"].median(), line_dash="dash", line_color="#e15759")
                    fig_hist.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title=selected, yaxis_title="Antal")
                    st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB: MEDICIN
# =============================================================================
with tabs[5]:
    st.markdown("### Medicin")
    
    pid_set = set(base["PID"].dropna().unique())
    med_raw = load_medicin()
    med = filter_to_cohort(med_raw, pid_set)
    
    if med.empty:
        st.info("Ingen medicin for denne kohorte.")
    else:
        med = filter_medicin(med)
        med["TIMESTAMP"] = pd.to_datetime(med["TIMESTAMP"])
        med = add_hours_from_start(med, "TIMESTAMP", base)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            hours_window = st.slider(
                "Tidsvindue (timer)", 
                min_value=1, max_value=168, value=24,
                key="med_hours"
            )
        
        med_w = med[
            (med["hours_from_start"] >= 0) & 
            (med["hours_from_start"] <= hours_window)
        ]
        
        if med_w.empty:
            st.info("Ingen data i dette tidsvindue.")
        else:
            # Category summary
            st.markdown("#### Kategorier")
            cat_summary = med_w.groupby("VALUE").agg(
                Administreringer=("PID", "count"),
                Patienter=("PID", "nunique")
            ).reset_index()
            cat_summary.columns = ["Kategori", "Administreringer", "Patienter"]
            cat_summary = cat_summary.sort_values("Patienter", ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(cat_summary, use_container_width=True, hide_index=True)
            with col2:
                fig = px.bar(
                    cat_summary.head(10).sort_values("Patienter"),
                    x="Patienter", y="Kategori", orientation="h",
                    color_discrete_sequence=["#4e79a7"]
                )
                fig.update_layout(
                    height=300, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Top preparations per category
            st.markdown("#### Top præparater pr. kategori")
            col1, col2 = st.columns([1, 3])
            with col1:
                top_n_med = st.slider("Antal", 5, 30, 15, key="med_topn")
            
            top_praep = med_w.groupby(["VALUE", "Generisk_navn"])["PID"].nunique().reset_index(name="Patienter")
            top_praep = top_praep.sort_values("Patienter", ascending=False).head(top_n_med)
            top_praep["label"] = top_praep["VALUE"] + " - " + top_praep["Generisk_navn"].str[:25]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_praep[["VALUE", "Generisk_navn", "Patienter"]], use_container_width=True, hide_index=True)
            with col2:
                fig = px.bar(
                    top_praep.sort_values("Patienter"),
                    x="Patienter", y="label", orientation="h",
                    color="VALUE",
                    color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
                )
                fig.update_layout(
                    height=max(300, top_n_med * 22),
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending"),
                    yaxis_title="",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Timeline per category
            st.markdown("#### Tidsprofil pr. kategori")
            
            med_w["time_bin"] = (med_w["hours_from_start"] // 2) * 2
            ts_med = med_w.groupby(["time_bin", "VALUE"])["PID"].nunique().reset_index(name="Patienter")
            
            fig_ts = px.line(
                ts_med, x="time_bin", y="Patienter", color="VALUE",
                markers=True,
                color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
            )
            fig_ts.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Timer fra start",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, title="")
            )
            st.plotly_chart(fig_ts, use_container_width=True)


# =============================================================================
# TAB: PROCEDURER
# =============================================================================
with tabs[6]:
    st.markdown("### Procedurer")
    
    pid_set = set(base["PID"].dropna().unique())
    proc_raw = load_procedurer()
    proc = filter_to_cohort(proc_raw, pid_set)
    
    if proc.empty:
        st.info("Ingen procedurer for denne kohorte.")
    else:
        proc = filter_procedures(proc)
        proc["TIMESTAMP"] = pd.to_datetime(proc["TIMESTAMP"])
        proc = add_hours_from_start(proc, "TIMESTAMP", base)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            hours_window = st.slider(
                "Tidsvindue (timer)", 
                min_value=1, max_value=168, value=24,
                key="proc_hours"
            )
        
        proc_w = proc[
            (proc["hours_from_start"] >= 0) & 
            (proc["hours_from_start"] <= hours_window)
        ]
        
        if proc_w.empty:
            st.info("Ingen data i dette tidsvindue.")
        else:
            # Category summary
            st.markdown("#### Kategorier")
            cat_summary = proc_w.groupby("VALUE").agg(
                Procedurer=("PID", "count"),
                Patienter=("PID", "nunique")
            ).reset_index()
            cat_summary.columns = ["Kategori", "Procedurer", "Patienter"]
            cat_summary = cat_summary.sort_values("Patienter", ascending=False)
            
            # Search for categories
            search_cat = st.text_input("Søg kategori", placeholder="Fx imaging, ventilation...", key="proc_cat_search")
            if search_cat:
                cat_summary = cat_summary[cat_summary["Kategori"].str.contains(search_cat, case=False, na=False)]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(cat_summary, use_container_width=True, hide_index=True)
            with col2:
                if not cat_summary.empty:
                    fig = px.bar(
                        cat_summary.head(10).sort_values("Patienter"),
                        x="Patienter", y="Kategori", orientation="h",
                        color_discrete_sequence=["#4e79a7"]
                    )
                    fig.update_layout(
                        height=300, margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(categoryorder="total ascending")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Top procedures by name
            st.markdown("#### Top procedurer pr. navn")
            col1, col2 = st.columns([1, 3])
            with col1:
                top_n_proc = st.slider("Antal", 5, 30, 15, key="proc_topn")
            
            top_proc_navn = proc_w.groupby(["VALUE", "ProcedureName"])["PID"].nunique().reset_index(name="Patienter")
            top_proc_navn = top_proc_navn.sort_values("Patienter", ascending=False)
            
            # Search for procedure name
            search_proc = st.text_input("Søg procedure", placeholder="Fx CT, ultralyd, intubation...", key="proc_name_search")
            if search_proc:
                top_proc_navn = top_proc_navn[top_proc_navn["ProcedureName"].str.contains(search_proc, case=False, na=False)]
            
            top_proc_navn_display = top_proc_navn.head(top_n_proc).copy()
            top_proc_navn_display["label"] = top_proc_navn_display["VALUE"] + " - " + top_proc_navn_display["ProcedureName"].str[:30]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_proc_navn_display[["VALUE", "ProcedureName", "Patienter"]], use_container_width=True, hide_index=True)
            with col2:
                if not top_proc_navn_display.empty:
                    fig = px.bar(
                        top_proc_navn_display.sort_values("Patienter"),
                        x="Patienter", y="label", orientation="h",
                        color="VALUE",
                        color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
                    )
                    fig.update_layout(
                        height=max(300, top_n_proc * 22),
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(categoryorder="total ascending"),
                        yaxis_title="",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Timeline per category
            st.markdown("#### Tidsprofil pr. kategori")
            
            proc_w["time_bin"] = (proc_w["hours_from_start"] // 2) * 2
            ts_proc = proc_w.groupby(["time_bin", "VALUE"])["PID"].nunique().reset_index(name="Patienter")
            
            fig_ts = px.line(
                ts_proc, x="time_bin", y="Patienter", color="VALUE",
                markers=True,
                color_discrete_sequence=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
            )
            fig_ts.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Timer fra start",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, title="")
            )
            st.plotly_chart(fig_ts, use_container_width=True)


# =============================================================================
# TAB: AFSNIT (ADT)
# =============================================================================
with tabs[7]:
    st.markdown("### Afsnit")
    
    pid_set = set(base["PID"].dropna().unique())
    adt_raw = load_adt()
    adt = filter_to_cohort(adt_raw, pid_set)
    
    if adt.empty:
        st.info("Ingen ADT-data for denne kohorte.")
    else:
        try:
            adt_filtered = filter_adt(adt, base_df=base)
        except:
            adt_filtered = adt.copy()
            if "Flyt_ind" in adt_filtered.columns:
                adt_filtered = adt_filtered.rename(columns={"Flyt_ind": "TIMESTAMP", "Flyt_ud": "END_TIMESTAMP"})
        
        for col in ["TIMESTAMP", "END_TIMESTAMP"]:
            if col in adt_filtered.columns:
                adt_filtered[col] = pd.to_datetime(adt_filtered[col], errors="coerce")
        
        type_col = "VALUE" if "VALUE" in adt_filtered.columns else None
        afsnit_col = "Afsnit" if "Afsnit" in adt_filtered.columns else None
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        n_afsnit = adt_filtered[afsnit_col].nunique() if afsnit_col else 0
        
        # Count unique ward episodes per patient
        adt_sorted = adt_filtered.sort_values(["PID", "TIMESTAMP"])
        if afsnit_col:
            adt_sorted["prev"] = adt_sorted.groupby("PID")[afsnit_col].shift()
            episodes = adt_sorted[adt_sorted[afsnit_col] != adt_sorted["prev"]]
            avg_episodes = episodes.groupby("PID").size().mean()
        else:
            avg_episodes = 0
        
        # TC reception
        tc_patients = 0
        if type_col and type_col in adt_filtered.columns:
            tc_patients = adt_filtered[adt_filtered[type_col].astype(str).str.upper() == "TC"]["PID"].nunique()
        
        with col1:
            st.metric("Unikke afsnit", n_afsnit)
        with col2:
            st.metric("Modtaget i TC", tc_patients)
        with col3:
            st.metric("Gns. afsnit pr. forløb", f"{avg_episodes:.1f}")
        
        st.markdown("---")
        
        # Distribution by type (TC, ICU, BED, OR, AMB)
        if type_col and type_col in adt_filtered.columns:
            st.markdown("#### Fordeling pr. afsnitstype")
            
            type_counts = adt_filtered.groupby(type_col)["PID"].nunique().reset_index()
            type_counts.columns = ["Type", "Patienter"]
            type_counts = type_counts.sort_values("Patienter", ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(type_counts, use_container_width=True, hide_index=True)
            with col2:
                fig = px.bar(
                    type_counts.sort_values("Patienter"),
                    x="Patienter", y="Type", orientation="h",
                    color_discrete_sequence=["#4e79a7"]
                )
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Duration stats
        if "END_TIMESTAMP" in adt_filtered.columns and "TIMESTAMP" in adt_filtered.columns:
            st.markdown("#### Varighed pr. afsnitstype")
            
            adt_filtered["duration_hours"] = (adt_filtered["END_TIMESTAMP"] - adt_filtered["TIMESTAMP"]).dt.total_seconds() / 3600
            adt_valid = adt_filtered[(adt_filtered["duration_hours"] >= 0) & (adt_filtered["duration_hours"].notna())]
            
            if type_col and not adt_valid.empty:
                dur_stats = adt_valid.groupby(type_col)["duration_hours"].agg(
                    Median="median",
                    Gennemsnit="mean",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75)
                ).round(1).reset_index()
                dur_stats.columns = ["Type", "Median (t)", "Gns (t)", "p25 (t)", "p75 (t)"]
                st.dataframe(dur_stats, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Top wards
        if afsnit_col:
            st.markdown("#### Top 10 afsnit")
            
            top_afsnit = adt_filtered.groupby(afsnit_col)["PID"].nunique().reset_index()
            top_afsnit.columns = ["Afsnit", "Patienter"]
            top_afsnit = top_afsnit.sort_values("Patienter", ascending=False).head(10)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_afsnit, use_container_width=True, hide_index=True)
            with col2:
                fig = px.bar(
                    top_afsnit.sort_values("Patienter"),
                    x="Patienter", y="Afsnit", orientation="h",
                    color_discrete_sequence=["#4e79a7"]
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Ward transitions
        st.markdown("#### Hyppigste overgange mellem afsnit")
        
        if afsnit_col:
            trans_df = adt_filtered.sort_values(["PID", "TIMESTAMP"]).copy()
            trans_df["next_afsnit"] = trans_df.groupby("PID")[afsnit_col].shift(-1)
            trans_df = trans_df.dropna(subset=["next_afsnit"])
            trans_df = trans_df[trans_df[afsnit_col] != trans_df["next_afsnit"]]
            
            trans_counts = trans_df.groupby([afsnit_col, "next_afsnit"]).size().reset_index(name="Antal")
            trans_counts = trans_counts.sort_values("Antal", ascending=False).head(15)
            trans_counts["Overgang"] = trans_counts[afsnit_col] + " -> " + trans_counts["next_afsnit"]
            
            st.dataframe(
                trans_counts[[afsnit_col, "next_afsnit", "Antal"]].rename(columns={afsnit_col: "Fra", "next_afsnit": "Til"}),
                use_container_width=True, 
                hide_index=True
            )


# =============================================================================
# TAB: DIAGNOSER
# =============================================================================
with tabs[8]:
    st.markdown("### Diagnoser")
    
    pid_set = set(base["PID"].dropna().unique())
    diag_raw = load_diagnoser()
    diag = filter_to_cohort(diag_raw, pid_set)
    
    if diag.empty:
        st.info("Ingen diagnoser for denne kohorte.")
    else:
        diag_col = next((c for c in ["Diagnosekode", "Diagnose", "diagnose", "DiagnoseKode"] if c in diag.columns), None)
        
        if diag_col:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            n_pat_diag = diag["PID"].nunique()
            n_diag_total = len(diag)
            diag_pr_pid = diag.groupby("PID").size()
            
            with col1:
                st.metric("Patienter med diagnoser", n_pat_diag)
            with col2:
                st.metric("Diagnoser i alt", format_number(n_diag_total))
            with col3:
                st.metric("Gns. pr. patient", f"{diag_pr_pid.mean():.1f}")
            with col4:
                st.metric("Median pr. patient", f"{diag_pr_pid.median():.0f}")
            
            st.markdown("---")
            
            # Top diagnoses
            st.markdown("#### Hyppigste diagnoser")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                top_n = st.slider("Antal diagnoser", 10, 50, 20, key="diag_topn")
            
            # Find tekst-kolonne og lav label med navn + kode
            text_col = next((c for c in ["Diagnose", "Diagnosetekst", "DiagnoseNavn", "Navn", "Tekst", "Beskrivelse"] if c in diag.columns and c != diag_col), None)
            if text_col:
                # Map kode → første forekommende tekst
                code_to_text = diag.dropna(subset=[text_col]).drop_duplicates(subset=[diag_col]).set_index(diag_col)[text_col].to_dict()
                diag["_diag_label"] = diag[diag_col].map(lambda c: f"{code_to_text.get(c, '')} ({c})" if code_to_text.get(c) else c)
                group_col = "_diag_label"
            else:
                group_col = diag_col

            top_diag = diag.groupby(group_col)["PID"].nunique().reset_index()
            top_diag.columns = ["Diagnose", "Patienter"]
            top_diag = top_diag.sort_values("Patienter", ascending=False)
            
            # Search
            search = st.text_input("Søg diagnose", placeholder="Fx S72, fraktur, pneumoni...", key="diag_search")
            if search:
                top_diag = top_diag[top_diag["Diagnose"].str.contains(search, case=False, na=False)]
            
            display_diag = top_diag.head(top_n)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(display_diag, use_container_width=True, hide_index=True, height=min(400, 35 + len(display_diag) * 35))
            with col2:
                fig = px.bar(
                    display_diag.head(15).sort_values("Patienter"),
                    x="Patienter", y="Diagnose", orientation="h",
                    color_discrete_sequence=["#4e79a7"]
                )
                fig.update_layout(
                    height=400, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ICD-10 chapter grouping
            st.markdown("#### ICD-10 kapitler")
            
            def icd_chapter(code):
                code = str(code).strip().upper()
                if not code:
                    return "Ukendt"
                # SKS-systemet: diagnoser har prefix 'D' (fx DA00 = ICD A00)
                if not code.startswith("D") or len(code) < 2:
                    return "Ukendt"
                c = code[1]
                chapters = {
                    "A": "Infektioner", "B": "Infektioner",
                    "C": "Neoplasmer", "D": "Neoplasmer",
                    "E": "Endokrine",
                    "F": "Psykiske",
                    "G": "Nervesystem",
                    "H": "Øje/Øre",
                    "I": "Kredsloeb",
                    "J": "Luftveje",
                    "K": "Mave-tarm",
                    "L": "Hud",
                    "M": "Bevæge",
                    "N": "Urinveje",
                    "O": "Graviditet",
                    "P": "Perinatal",
                    "Q": "Medfødte",
                    "R": "Symptomer",
                    "S": "Skader", "T": "Skader",
                    "V": "Ydre", "W": "Ydre", "X": "Ydre", "Y": "Ydre",
                    "Z": "Kontakt",
                }
                return chapters.get(c, "Andet")
            
            diag["Kapitel"] = diag[diag_col].apply(icd_chapter)
            chapter_counts = diag.groupby("Kapitel")["PID"].nunique().reset_index()
            chapter_counts.columns = ["Kapitel", "Patienter"]
            chapter_counts = chapter_counts.sort_values("Patienter", ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(chapter_counts, use_container_width=True, hide_index=True)
            with col2:
                fig_ch = px.bar(
                    chapter_counts.sort_values("Patienter"),
                    x="Patienter", y="Kapitel", orientation="h",
                    color_discrete_sequence=["#4e79a7"]
                )
                fig_ch.update_layout(
                    height=350, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(categoryorder="total ascending")
                )
                st.plotly_chart(fig_ch, use_container_width=True)
        else:
            st.error("Kunne ikke finde diagnosekolonne i data.")