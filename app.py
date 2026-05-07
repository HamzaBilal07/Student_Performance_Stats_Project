from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import dataset_overview, load_default_dataset
from src.distributions import (
    binomial_distribution,
    hypergeometric_distribution,
    normal_fit,
    poisson_distribution,
)
from src.eda import (
    bar_chart,
    box_plot,
    categorical_summary,
    correlation_heatmap,
    covariance_correlation,
    descriptive_statistics,
    histogram,
    pie_chart,
    save_figure,
    scatter_plot,
)
from src.inference import (
    confidence_interval_mean,
    one_sample_t_test,
    p_value_conclusion,
    two_sample_t_test,
)
from src.preprocessing import clean_dataset
from src.probability import (
    bayes_theorem,
    conditional_probability,
    joint_probability,
    probability_value_options,
    simple_probability,
)
from src.regression import predict_student_performance, train_regression_model
from src.report_generator import generate_report
from src.supplementary_r_syntax import R_SYNTAX_REFERENCES, get_r_syntax


LOGO_PATH = PROJECT_ROOT / "assets" / "logo_placeholder.png"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROJECT_TITLE = "Statistical Analysis and Prediction of Student Academic Performance Using Python"
TEAM_NAME = "HAATS Academic Analytics"
TEAM_MEMBERS = [
    {
        "Roll Number": "24F-0569",
        "Name": "Muhammad Hamza Bilal",
        "Section": "BS(CS)4E",
        "Role": "Group Leader",
    },
    {
        "Roll Number": "24F-0563",
        "Name": "Aytsamullah",
        "Section": "BS(CS)4E",
        "Role": "Member",
    },
    {
        "Roll Number": "24F-0577",
        "Name": "Ali Haider",
        "Section": "BS(CS)4E",
        "Role": "Member",
    },
    {
        "Roll Number": "24F-3085",
        "Name": "Talha Asif",
        "Section": "BS(SE) 4B",
        "Role": "Member",
    },
    {
        "Roll Number": "24F-3104",
        "Name": "Muhammad Shahab Raheem",
        "Section": "BS(SE) 4B",
        "Role": "Member",
    },
]


st.set_page_config(
    page_title="Student Performance Stats Project",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #172033;
        --muted: #53657d;
        --line: #d8e2ec;
        --paper: #ffffff;
        --soft: #f4f8fb;
        --blue: #2f6f9f;
        --teal: #1b8a84;
        --gold: #d9a441;
        --sage: #e8f3ef;
        --sky: #edf5fb;
        --shadow: 0 14px 34px rgba(23, 32, 51, 0.085);
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(27,138,132,0.12), transparent 28rem),
            radial-gradient(circle at top right, rgba(47,111,159,0.10), transparent 32rem),
            linear-gradient(180deg, #f7fafb 0%, #eef4f7 46%, #f7fafb 100%);
    }
    .main {
        background: transparent;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1360px;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.72rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    h2, h3 {
        font-weight: 800;
    }
    .hero {
        background:
            linear-gradient(135deg, #172033 0%, #24466d 52%, #1b8a84 100%);
        color: white;
        border-radius: 8px;
        padding: 2rem 2.1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 22px 50px rgba(23, 32, 51, 0.20);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -5rem;
        top: -8rem;
        width: 20rem;
        height: 20rem;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
    }
    .hero h1 {
        color: white;
        font-size: clamp(1.55rem, 2.8vw, 2.45rem);
        margin-bottom: 0.4rem;
        line-height: 1.12;
    }
    .hero p {
        color: #dcebf4;
        font-size: 1.05rem;
        margin-bottom: 0;
        max-width: 68rem;
    }
    .card {
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
        border: 1px solid var(--line);
        border-top: 4px solid var(--teal);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow);
        min-height: 112px;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 36px rgba(23, 32, 51, 0.12);
    }
    .metric-card .label {
        color: #607086;
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    .metric-card .value {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-card .detail {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .section-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f6fbfa 100%);
        border: 1px solid var(--line);
        border-left: 5px solid var(--teal);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 0.8rem;
        box-shadow: 0 10px 24px rgba(23, 32, 51, 0.055);
    }
    .section-banner h3 {
        margin: 0 0 0.25rem 0;
        font-size: 1.12rem;
    }
    .section-banner p {
        margin: 0;
        color: var(--muted);
    }
    .team-pill {
        display: inline-block;
        background: rgba(217,164,65,0.14);
        color: #6b4a11;
        border: 1px solid rgba(217,164,65,0.36);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-weight: 700;
        margin: 0.15rem 0.2rem 0.15rem 0;
    }
    .explain {
        background: #ffffff;
        color: #34465c;
        border: 1px solid #dbe3ef;
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 18px rgba(23, 32, 51, 0.045);
    }
    .warn {
        background: #fff9eb;
        color: #6b4a11;
        border: 1px solid #edd49a;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0;
    }
    div[data-testid="stSidebar"],
    div[data-testid="stSidebarContent"] {
        background:
            linear-gradient(180deg, #111827 0%, #172033 54%, #173f46 100%);
    }
    div[data-testid="stSidebarContent"] {
        padding: 1.25rem 0.75rem;
    }
    div[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    .sidebar-logo-link {
        display: block;
        text-decoration: none !important;
        margin: 0.25rem auto 0.9rem;
        max-width: 230px;
    }
    .sidebar-logo-link img {
        width: 100%;
        display: block;
        border-radius: 8px;
        background: #f8fbfc;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 14px 28px rgba(0,0,0,0.22);
        transition: transform 150ms ease, box-shadow 150ms ease;
    }
    .sidebar-logo-link:hover img {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(0,0,0,0.28);
    }
    .sidebar-kicker {
        margin: 0.35rem 0 0.25rem;
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .sidebar-caption {
        margin: 0 0 0.8rem;
        color: #a6ded8 !important;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .nav-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.46rem;
        margin-top: 0.9rem;
    }
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        padding: 0.62rem 0.66rem;
        border-radius: 8px;
        text-decoration: none !important;
        color: #2e3b4f !important;
        border: 1px solid rgba(125, 143, 166, 0.38);
        background: rgba(248, 251, 252, 0.92);
        box-shadow: 0 8px 18px rgba(0,0,0,0.10);
        transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
    }
    .nav-item:hover {
        transform: translateX(3px);
        background: #ffffff;
        border-color: rgba(27, 138, 132, 0.58);
        color: var(--ink) !important;
    }
    .nav-item.active {
        background: linear-gradient(135deg, #ffffff 0%, #e8f6f3 100%);
        border-color: rgba(217, 164, 65, 0.65);
        color: var(--ink) !important;
        box-shadow: 0 14px 30px rgba(0,0,0,0.24);
    }
    .nav-number {
        flex: 0 0 1.7rem;
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #4d6077 !important;
        background: #e5edf2;
        border: 1px solid rgba(83, 101, 125, 0.24);
        font-size: 0.78rem;
        font-weight: 900;
    }
    .nav-title {
        color: inherit !important;
        font-size: 0.92rem;
        font-weight: 750;
        line-height: 1.18;
    }
    .nav-item.active .nav-number {
        color: #ffffff !important;
        background: linear-gradient(135deg, #1b8a84, #2f6f9f);
        border-color: transparent;
    }
    div.stButton > button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 8px;
        border: 1px solid #236184;
        background: linear-gradient(135deg, #2f6f9f, #1b8a84);
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 18px rgba(47, 111, 159, 0.18);
    }
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: #1b8a84;
        color: white;
        filter: brightness(1.04);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    textarea,
    input {
        border-radius: 8px !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label {
        color: #34465c !important;
        font-weight: 750 !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: 0 10px 24px rgba(23, 32, 51, 0.055);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px 8px 0 0;
        padding: 0.45rem 0.8rem;
    }
    pre {
        border-radius: 8px !important;
        border: 1px solid #dbe3ef !important;
    }
    div.streamlit-expanderHeader {
        font-weight: 800;
        color: var(--ink);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


PAGE_ITEMS = [
    ("home", "1", "Home / Title Page"),
    ("dataset", "2", "Dataset Overview"),
    ("cleaning", "3", "Data Cleaning"),
    ("descriptive", "4", "Descriptive Statistics"),
    ("eda", "5", "EDA Visualizations"),
    ("probability", "6", "Probability Analysis"),
    ("distributions", "7", "Probability Distributions"),
    ("inference", "8", "Confidence Intervals & Hypothesis Testing"),
    ("regression", "9", "Regression & Prediction"),
    ("summary", "10", "Final Results Summary"),
    ("report", "11", "Report Generator"),
]

PAGE_KEYS = {key for key, _, _ in PAGE_ITEMS}


@st.cache_data(show_spinner=False)
def cached_default_dataset() -> tuple[pd.DataFrame, str]:
    return load_default_dataset()


@st.cache_data(show_spinner=False)
def cached_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    result = clean_dataset(df)
    return result.cleaned_df, result.metadata


def dataset_signature(df: pd.DataFrame) -> str:
    content_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    columns = "|".join(df.columns.astype(str))
    return f"{df.shape[0]}x{df.shape[1]}::{columns}::{content_hash}"


def initialize_state() -> None:
    with st.spinner("Loading project dataset..."):
        df, source = cached_default_dataset()
    signature = dataset_signature(df)
    if st.session_state.get("dataset_signature") != signature:
        st.session_state.dataset_signature = signature
        st.session_state.graph_paths = []
        st.session_state.regression_result = None
    st.session_state.raw_df = df
    st.session_state.dataset_source = source
    st.session_state.setdefault("graph_paths", [])
    st.session_state.setdefault("regression_result", None)


def get_clean_data() -> tuple[pd.DataFrame, dict[str, object]]:
    return cached_clean_dataset(st.session_state.raw_df)


def register_graph(path: str) -> None:
    graph_paths = st.session_state.setdefault("graph_paths", [])
    if path not in graph_paths:
        graph_paths.append(path)


def show_figure(fig, filename: str) -> str:
    path = save_figure(fig, filename)
    register_graph(path)
    st.pyplot(fig)
    return path


def card(label: str, value: object, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def explain(text: str) -> None:
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn">{text}</div>', unsafe_allow_html=True)


def section_banner(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-banner">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def supplementary_r_syntax(title: str, topic: str, expanded: bool = False) -> None:
    code = get_r_syntax(topic)
    if not code:
        return
    with st.expander(f"Supplementary R Syntax: {title}", expanded=expanded):
        st.caption("Syntax reference for the corresponding statistical method.")
        st.code(code, language="r")


def default_target_column(numeric_cols: list[str]) -> str | None:
    preferred = [
        "Exam_Score",
        "Final_Score",
        "Performance_Score",
        "Score",
        "GPA",
        "Grade",
        "Marks",
    ]
    lower_map = {col.lower(): col for col in numeric_cols}
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return numeric_cols[-1] if numeric_cols else None


def default_predictors(df: pd.DataFrame, target: str, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    preferred = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
        "Motivation_Level",
        "Internet_Access",
        "Family_Income",
    ]
    predictors = [col for col in preferred if col in df.columns and col != target]
    if predictors:
        return predictors
    predictor_candidates = [col for col in numeric_cols + categorical_cols if col != target]
    return predictor_candidates[: min(8, len(predictor_candidates))]


def title_block(subtitle: str = "Student Performance & Lifestyle Analysis") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{PROJECT_TITLE}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def selected_page_key() -> str:
    page = st.query_params.get("page", "home")
    if isinstance(page, list):
        page = page[0] if page else "home"
    return page if page in PAGE_KEYS else "home"


def sidebar_navigation() -> str:
    with st.sidebar:
        logo_src = logo_data_uri()
        if logo_src:
            st.markdown(
                f'<a class="sidebar-logo-link" href="?page=home" target="_self" title="Home">'
                f'<img src="{logo_src}" alt="HAATS Academic Analytics logo"></a>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<a class="sidebar-logo-link" href="?page=home" target="_self">Home</a>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-kicker">Probability & Statistics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">Semester Project</div>', unsafe_allow_html=True)

        active_key = selected_page_key()
        links = []
        for key, number, title in PAGE_ITEMS:
            active_class = " active" if key == active_key else ""
            links.append(
                f'<a class="nav-item{active_class}" href="?page={key}" target="_self">'
                f'<span class="nav-number">{number}</span>'
                f'<span class="nav-title">{title}</span>'
                f'</a>'
            )
        st.markdown(f'<nav class="nav-wrap">{"".join(links)}</nav>', unsafe_allow_html=True)
        return active_key


def members_placeholder() -> pd.DataFrame:
    return pd.DataFrame(TEAM_MEMBERS)


def page_home() -> None:
    title_block()
    left, right = st.columns([1, 3])
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=360)
        else:
            st.info("Team logo")
    with right:
        st.markdown("### Team Name")
        st.markdown(f"**{TEAM_NAME}**")
        st.markdown(
            """
            <span class="team-pill">Leader: Muhammad Hamza Bilal</span>
            <span class="team-pill">BS(CS)4E</span>
            <span class="team-pill">BS(SE) 4B</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Group Members")
        st.dataframe(members_placeholder(), use_container_width=True, hide_index=True)

    section_banner(
        "Project Snapshot",
        "Statistical analysis, predictive modeling, and formal report generation for student performance data.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Dataset", "Project CSV", "Student performance factors dataset")
    with c2:
        card("Core Method", "Statistics", "EDA, probability, inference, regression")
    with c3:
        card("Prediction", "Regression", "Final score/performance prediction")
    with c4:
        card("Report", ".docx", "Word report with graphs and results")

    section_banner(
        "Course Coverage",
        "Core Probability and Statistics concepts covered in the project analysis.",
    )
    coverage = pd.DataFrame(
        {
            "Area": [
                "Data Types",
                "Descriptive Statistics",
                "EDA Graphs",
                "Probability",
                "Distributions",
                "Inference",
                "Regression",
                "Report",
            ],
            "Included Concepts": [
                "Qualitative, quantitative, discrete, continuous",
                "Mean, median, mode, quartiles, percentiles, deciles, variance, standard deviation, IQR",
                "Tables, bar chart, pie chart, histogram, box plot, scatter plot, correlation heatmap",
                "P(A), P(A and B), P(A|B), Bayes theorem",
                "Binomial, Poisson, Normal distribution",
                "Confidence interval, one-sample t-test, two-sample t-test, p-value conclusion",
                "OLS, simple and multiple linear regression, model evaluation, prediction form",
                "Word report generated with python-docx and saved graphs",
            ],
        }
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    supplementary_r_syntax("Summary statistics syntax", "summary")


def page_dataset_overview() -> None:
    title_block("Dataset Overview")
    section_banner(
        "Project Dataset",
        "All statistical analysis, prediction, and report generation are based on the Student Performance Factors dataset.",
    )
    st.write(st.session_state.dataset_source)
    st.caption("Dataset file: data/student_performance_factors.csv")

    df = st.session_state.raw_df
    overview = dataset_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Rows", f"{overview['rows']:,}", "Student records")
    with c2:
        card("Columns", overview["columns"], "Variables")
    with c3:
        card("Numerical", len(overview["numerical_columns"]), "Quantitative variables")
    with c4:
        card("Categorical", len(overview["categorical_columns"]), "Qualitative variables")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Column Overview")
    column_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Pandas Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isna().sum().values,
            "Unique Values": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    st.dataframe(column_table, use_container_width=True, hide_index=True)

    explain(
        "Sample vs population: the displayed records are the sample used for analysis. "
        "A population would be every student in the university, school system, or target academic group."
    )


def page_cleaning() -> None:
    title_block("Automatic Data Cleaning")
    df, metadata = get_clean_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Original Shape", f"{metadata['raw_shape'][0]} x {metadata['raw_shape'][1]}", "Rows x columns")
    with c2:
        card("Cleaned Shape", f"{metadata['cleaned_shape'][0]} x {metadata['cleaned_shape'][1]}", "After cleaning")
    with c3:
        card("Duplicates Removed", metadata["duplicate_rows_removed"], "Exact duplicate rows")
    with c4:
        card("Remaining Missing", int(df.isna().sum().sum()), "After imputation")

    st.markdown("### Cleaning Steps")
    for step in metadata["cleaning_steps"]:
        st.write(f"- {step}")

    st.markdown("### Missing Values Before Cleaning")
    st.dataframe(metadata["missing_before"], use_container_width=True, hide_index=True)

    st.markdown("### Missing Values After Cleaning")
    st.dataframe(metadata["missing_after"], use_container_width=True, hide_index=True)

    st.markdown("### Detected Course Data Types")
    st.dataframe(metadata["type_table"], use_container_width=True, hide_index=True)

    section_banner(
        "Variable Classification",
        "Qualitative variables describe categories. Quantitative variables are numerical; discrete values are count-like, while continuous values vary over an interval.",
    )


def page_descriptive_statistics() -> None:
    title_block("Descriptive Statistics")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]

    if not numeric_cols:
        st.warning("No numerical columns were detected.")
        return

    selected = st.multiselect(
        "Select numerical columns",
        numeric_cols,
        default=numeric_cols[: min(6, len(numeric_cols))],
    )
    if selected:
        section_banner(
            "Numerical Summary Table",
            "Count, mean, median, mode, spread, quartiles, percentiles, and deciles for selected variables.",
        )
        stats_table = descriptive_statistics(df, selected)
        st.dataframe(stats_table, use_container_width=True, hide_index=True)
        explain(
            "Interpretation: the mean gives the arithmetic average, the median gives the middle value, "
            "and the mode gives the most frequent value. The IQR measures the spread of the middle 50 percent of values."
        )
        supplementary_r_syntax("Summary statistics", "summary")

    if categorical_cols:
        section_banner("Categorical Summary", "Frequency-style summary for qualitative variables.")
        st.dataframe(categorical_summary(df, categorical_cols), use_container_width=True, hide_index=True)

    if len(selected) >= 2:
        section_banner(
            "Covariance and Correlation",
            "Covariance measures joint movement; correlation standardizes it between -1 and +1.",
        )
        cov, corr = covariance_correlation(df, selected)
        left, right = st.columns(2)
        with left:
            st.caption("Covariance Matrix")
            st.dataframe(cov, use_container_width=True)
        with right:
            st.caption("Correlation Matrix")
            st.dataframe(corr, use_container_width=True)
        explain(
            "Covariance shows whether two variables move together, while correlation standardizes that movement "
            "between -1 and +1 so relationships are easier to compare."
        )
        supplementary_r_syntax("Covariance and correlation", "covariance_correlation")
        supplementary_r_syntax("Manual correlation formula", "correlation_manual")


def page_eda_visualizations() -> None:
    title_block("Exploratory Data Analysis Visualizations")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]

    section_banner("Dataset Preview", "Initial records from the Student Performance Factors dataset.")
    st.dataframe(df.head(12), use_container_width=True)

    if categorical_cols:
        section_banner("Categorical Graphs", "Bar and pie charts show category frequencies and proportions.")
        left, right = st.columns(2)
        with left:
            bar_col = st.selectbox("Bar chart categorical column", categorical_cols, key="bar_col")
            fig = bar_chart(df, bar_col)
            show_figure(fig, f"bar_chart_{bar_col}.png")
        with right:
            pie_col = st.selectbox("Pie chart categorical column", categorical_cols, key="pie_col")
            fig = pie_chart(df, pie_col)
            show_figure(fig, f"pie_chart_{pie_col}.png")
    else:
        warn("No categorical columns were detected for bar and pie charts.")

    if numeric_cols:
        section_banner("Numerical Graphs", "Histograms show distribution shape; box plots reveal spread and possible outliers.")
        left, right = st.columns(2)
        with left:
            hist_col = st.selectbox("Histogram numerical column", numeric_cols, key="hist_col")
            fig = histogram(df, hist_col)
            show_figure(fig, f"histogram_{hist_col}.png")
        with right:
            box_col = st.selectbox("Box plot numerical column", numeric_cols, key="box_col")
            group_choice = None
            if categorical_cols:
                group_options = ["No grouping"] + categorical_cols
                selected_group = st.selectbox("Optional categorical grouping", group_options)
                group_choice = None if selected_group == "No grouping" else selected_group
            fig = box_plot(df, box_col, group_choice)
            show_figure(fig, f"box_plot_{box_col}.png")

    if len(numeric_cols) >= 2:
        section_banner("Scatter Plot", "Compare two numerical variables and optionally color by a categorical group.")
        left, middle, right = st.columns(3)
        with left:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
        with middle:
            y_default = numeric_cols.index(default_target_column(numeric_cols)) if default_target_column(numeric_cols) in numeric_cols else 1
            y_col = st.selectbox("Y-axis", numeric_cols, index=min(y_default, len(numeric_cols) - 1))
        with right:
            hue_options = ["None"] + categorical_cols
            hue = st.selectbox("Color grouping", hue_options)
        fig = scatter_plot(df, x_col, y_col, None if hue == "None" else hue)
        show_figure(fig, f"scatter_{x_col}_vs_{y_col}.png")

        section_banner("Correlation Heatmap", "A compact visual matrix of pairwise numerical relationships.")
        selected_heatmap_cols = st.multiselect(
            "Columns for heatmap",
            numeric_cols,
            default=numeric_cols[: min(7, len(numeric_cols))],
        )
        if len(selected_heatmap_cols) >= 2:
            fig = correlation_heatmap(df, selected_heatmap_cols)
            show_figure(fig, "correlation_heatmap.png")

    supplementary_r_syntax("Frequency tables and graphical representation", "frequency_graphs")


def probability_dataframe(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    prob_df = df.copy()
    usable_cols = list(categorical_cols)
    if len(usable_cols) < 2:
        for column in numeric_cols[:3]:
            new_col = f"{column}_Band"
            try:
                prob_df[new_col] = pd.qcut(prob_df[column], q=3, labels=["Low", "Medium", "High"], duplicates="drop").astype(str)
                usable_cols.append(new_col)
            except Exception:
                continue
    return prob_df, usable_cols


def page_probability_analysis() -> None:
    title_block("Probability Analysis")
    df, metadata = get_clean_data()
    prob_df, categorical_cols = probability_dataframe(
        df, metadata["numerical_columns"], metadata["categorical_columns"]
    )

    if len(categorical_cols) < 2:
        st.warning("At least two categorical or binned columns are needed for probability analysis.")
        return

    section_banner(
        "Select Events",
        "Choose two categorical conditions from the dataset; numerical variables are binned automatically when needed.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        col_a = st.selectbox("Event A column", categorical_cols, key="prob_a_col")
    with c2:
        value_a = st.selectbox("Event A value", probability_value_options(prob_df, col_a), key="prob_a_val")
    with c3:
        col_b = st.selectbox("Event B column", categorical_cols, key="prob_b_col")
    with c4:
        value_b = st.selectbox("Event B value", probability_value_options(prob_df, col_b), key="prob_b_val")

    p_a = simple_probability(prob_df, col_a, value_a)
    p_b = simple_probability(prob_df, col_b, value_b)
    p_joint = joint_probability(prob_df, col_a, value_a, col_b, value_b)
    p_cond = conditional_probability(prob_df, col_a, value_a, col_b, value_b)
    bayes = bayes_theorem(prob_df, col_a, value_a, col_b, value_b)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("P(A)", f"{p_a['probability']:.4f}", f"{p_a['count']} of {p_a['total']} rows")
    with c2:
        card("P(B)", f"{p_b['probability']:.4f}", f"{p_b['count']} of {p_b['total']} rows")
    with c3:
        card("P(A and B)", f"{p_joint['probability']:.4f}", f"{p_joint['count']} joint rows")
    with c4:
        card("P(A | B)", f"{p_cond['probability']:.4f}", f"{p_cond['count_a_and_b']} of {p_cond['count_b']} B rows")

    section_banner("Bayes Theorem", "Use P(A), P(B), and P(B|A) to calculate or verify P(A|B).")
    st.latex(r"P(A|B)=\frac{P(B|A)P(A)}{P(B)}")
    st.dataframe(pd.DataFrame([bayes]).round(5), use_container_width=True, hide_index=True)
    explain(
        f"Here, A means {col_a} = {value_a}, and B means {col_b} = {value_b}. "
        "Bayes theorem lets us reverse a conditional probability when P(B|A), P(A), and P(B) are known."
    )
    supplementary_r_syntax("Conditional probability using cross tab", "conditional_probability")
    supplementary_r_syntax("Bayes theorem", "bayes")


def page_distributions() -> None:
    title_block("Probability Distributions")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]

    section_banner("Binomial Distribution", "Models the number of successes in a set number of independent trials.")
    c1, c2 = st.columns([1, 2])
    with c1:
        n_trials = st.slider("Number of trials n", 1, 100, 20)
        probability_success = st.slider("Probability of success p", 0.01, 0.99, 0.60)
        k_value = st.number_input("Successes for probability lookup", 0, n_trials, min(10, n_trials))
    table, fig = binomial_distribution(n_trials, probability_success)
    with c2:
        show_figure(fig, "binomial_distribution.png")
    lookup = table.loc[table["Successes"] == k_value, "Probability"].iloc[0]
    st.write(f"Probability of exactly {k_value} successes: **{lookup:.5f}**")
    explain("A binomial distribution models a set number of independent trials where each trial has success or failure.")
    supplementary_r_syntax("Binomial distribution", "binomial")

    section_banner("Poisson Distribution", "Models event counts when the average rate of occurrence is known.")
    c1, c2 = st.columns([1, 2])
    with c1:
        lambda_rate = st.number_input("Average event rate lambda", min_value=0.10, max_value=50.0, value=3.0, step=0.1)
        max_k = st.number_input("Maximum events shown", min_value=5, max_value=100, value=15)
    table, fig = poisson_distribution(lambda_rate, int(max_k))
    with c2:
        show_figure(fig, "poisson_distribution.png")
    explain("A Poisson distribution models the number of events occurring in a fixed interval when the average rate is known.")
    supplementary_r_syntax("Poisson distribution", "poisson")

    section_banner("Hypergeometric Distribution", "Models successes drawn without replacement from a finite population.")
    c1, c2 = st.columns([1, 2])
    with c1:
        population_size = st.number_input("Population size N", min_value=2, max_value=500, value=10)
        success_states = st.number_input("Success states K", min_value=1, max_value=int(population_size), value=min(5, int(population_size)))
        draws = st.number_input("Draws n", min_value=1, max_value=int(population_size), value=min(6, int(population_size)))
        min_x = max(0, int(draws) - (int(population_size) - int(success_states)))
        max_x = min(int(success_states), int(draws))
        hyper_x = st.number_input("Successes x for lookup", min_value=min_x, max_value=max_x, value=min(max(3, min_x), max_x))
    hyper_table, hyper_fig = hypergeometric_distribution(
        int(population_size), int(success_states), int(draws)
    )
    with c2:
        show_figure(hyper_fig, "hypergeometric_distribution.png")
    hyper_lookup = hyper_table.loc[hyper_table["Successes"] == hyper_x, "Probability"]
    if not hyper_lookup.empty:
        st.write(f"Probability of exactly {hyper_x} successes: **{hyper_lookup.iloc[0]:.5f}**")
    explain(
        "Hypergeometric probability is useful when selections are dependent because items are not replaced after each draw."
    )
    supplementary_r_syntax("Hypergeometric distribution", "hypergeometric")

    section_banner("Normal Distribution Fit", "Overlay a bell curve on a selected numerical student-performance variable.")
    if numeric_cols:
        selected_col = st.selectbox("Select numerical column for normal fit", numeric_cols)
        stats_info, fig = normal_fit(df[selected_col], selected_col)
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            card("Mean", f"{stats_info['mean']:.3f}", "Center of distribution")
        with c2:
            card("Std Dev", f"{stats_info['std_dev']:.3f}", "Average spread")
        with c3:
            show_figure(fig, f"normal_fit_{selected_col}.png")
        explain("The normal curve helps compare the selected variable with the bell-shaped distribution assumption.")


def page_inference() -> None:
    title_block("Confidence Intervals & Hypothesis Testing")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]

    if not numeric_cols:
        st.warning("No numerical columns were detected for inference.")
        return

    selected_col = st.selectbox("Numerical variable", numeric_cols, index=0)
    confidence_label = st.selectbox("Confidence level", ["90%", "95%", "99%"], index=1)
    confidence_level = {"90%": 0.90, "95%": 0.95, "99%": 0.99}[confidence_label]
    alpha = 1 - confidence_level

    ci = confidence_interval_mean(df[selected_col], confidence_level)
    st.markdown("### Confidence Interval for Mean")
    st.dataframe(pd.DataFrame([ci]).round(4), use_container_width=True, hide_index=True)
    explain(
        f"At the {confidence_label} confidence level, the interval estimates the likely range of the population mean for {selected_col}."
    )

    st.markdown("### One-Sample t-test")
    default_mean = float(pd.to_numeric(df[selected_col], errors="coerce").mean())
    hypothesized_mean = st.number_input(
        "Hypothesized population mean",
        value=round(default_mean, 2),
        step=1.0,
    )
    one_test = one_sample_t_test(df[selected_col], hypothesized_mean)
    st.write(f"Null hypothesis: {one_test['null_hypothesis']}")
    st.write(f"Alternative hypothesis: {one_test['alternative_hypothesis']}")
    c1, c2 = st.columns(2)
    with c1:
        card("Test Statistic", f"{one_test['test_statistic']:.4f}", "t-value")
    with c2:
        card("p-value", f"{one_test['p_value']:.4f}", "Decision evidence")
    explain(p_value_conclusion(one_test["p_value"], alpha))

    st.markdown("### Two-Sample t-test by Group")
    if not categorical_cols:
        warn("No categorical grouping variable was detected for the two-sample t-test.")
        return

    group_col = st.selectbox("Categorical grouping variable", categorical_cols)
    group_values = df[group_col].astype(str).value_counts().head(20).index.tolist()
    if len(group_values) < 2:
        warn("The selected grouping variable needs at least two groups.")
        return

    left, right = st.columns(2)
    with left:
        group_a = st.selectbox("Group A", group_values, index=0)
    with right:
        group_b = st.selectbox("Group B", group_values, index=1)

    if group_a == group_b:
        warn("Select two different groups.")
        return

    two_test = two_sample_t_test(df, selected_col, group_col, group_a, group_b)
    st.write(f"Null hypothesis: {two_test['null_hypothesis']}")
    st.write(f"Alternative hypothesis: {two_test['alternative_hypothesis']}")
    st.dataframe(pd.DataFrame([two_test]).round(4), use_container_width=True, hide_index=True)
    explain(p_value_conclusion(two_test["p_value"], alpha))


def display_regression_result(result: dict[str, object]) -> None:
    section_banner(
        str(result["model_type"]),
        "Model results, coefficient table, OLS summary, and actual-vs-predicted graph.",
    )
    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("R-squared", f"{metrics['R-squared']:.4f}", "Explained variance")
    with c2:
        card("MAE", f"{metrics['MAE']:.4f}", "Average absolute error")
    with c3:
        card("MSE", f"{metrics['MSE']:.4f}", "Squared error")
    with c4:
        card("RMSE", f"{metrics['RMSE']:.4f}", "Error in score units")

    st.write(f"Intercept: **{result['intercept']:.4f}**")
    section_banner("Regression Coefficients", "Each coefficient estimates how the predictor changes the target score.")
    st.dataframe(result["coefficients"], use_container_width=True, hide_index=True)

    section_banner("Actual vs Predicted", "Points closer to the red diagonal line indicate more accurate predictions.")
    show_figure(result["actual_predicted_fig"], f"actual_vs_predicted_{result['target']}.png")

    with st.expander("OLS statistical summary"):
        st.text(result["ols_summary"])

    explain(
        "Regression interpretation: positive coefficients increase the predicted target when other predictors are held constant. "
        "Negative coefficients decrease it. R-squared tells how much target variation the model explains."
    )


def page_regression() -> None:
    title_block("Regression & Prediction")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]

    if not numeric_cols:
        st.warning("Regression requires a numerical target variable.")
        return

    target_default = default_target_column(numeric_cols)
    target_index = numeric_cols.index(target_default) if target_default in numeric_cols else 0
    section_banner(
        "Model Setup",
        "Choose a numerical target and any mix of numerical/categorical predictors; categorical variables are encoded automatically.",
    )
    target = st.selectbox("Target numerical variable", numeric_cols, index=target_index)
    predictor_options = [col for col in df.columns if col != target]
    selected_predictors = st.multiselect(
        "Predictor variables",
        predictor_options,
        default=default_predictors(df, target, numeric_cols, categorical_cols),
    )
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, step=0.05)

    if st.button("Train Regression Model", type="primary"):
        try:
            result = train_regression_model(df, target, selected_predictors, test_size=test_size)
            st.session_state.regression_result = result
            st.success("Regression model trained successfully.")
        except Exception as exc:
            st.error(f"Could not train model: {exc}")

    result = st.session_state.get("regression_result")
    if result and result.get("target") == target and result.get("predictors") == selected_predictors:
        display_regression_result(result)

        st.markdown("### Prediction Form")
        with st.form("prediction_form"):
            input_values: dict[str, object] = {}
            for predictor in selected_predictors:
                if predictor in numeric_cols:
                    values = pd.to_numeric(df[predictor], errors="coerce")
                    input_values[predictor] = st.number_input(
                        predictor,
                        value=float(values.mean()),
                        min_value=float(values.min()),
                        max_value=float(values.max()),
                    )
                else:
                    options = df[predictor].astype(str).value_counts().index.tolist()
                    input_values[predictor] = st.selectbox(predictor, options)
            submitted = st.form_submit_button("Predict Student Performance")

        if submitted:
            prediction = predict_student_performance(result["pipeline"], input_values)
            card("Predicted Performance", f"{prediction:.2f}", f"Target: {target}")
    else:
        warn("Train the regression model before generating predictions.")

    supplementary_r_syntax("Manual regression formulas", "regression_manual")


def build_results_summary(
    df: pd.DataFrame,
    metadata: dict[str, object],
    target: str | None,
    regression_result: dict[str, object] | None,
) -> list[str]:
    summary = [
        f"The cleaned dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} variables.",
        f"Detected {len(metadata['numerical_columns'])} quantitative and {len(metadata['categorical_columns'])} qualitative variables.",
        f"Removed {metadata['duplicate_rows_removed']} duplicate rows and imputed missing values.",
    ]
    if target and target in df.columns:
        target_values = pd.to_numeric(df[target], errors="coerce")
        summary.append(
            f"The mean {target} is {target_values.mean():.2f}, with standard deviation {target_values.std(ddof=1):.2f}."
        )
        numeric_cols = [col for col in metadata["numerical_columns"] if col != target]
        if numeric_cols:
            corr = df[numeric_cols + [target]].corr()[target].drop(target).sort_values(key=lambda values: values.abs(), ascending=False)
            if not corr.empty:
                summary.append(f"The strongest numerical relationship with {target} is {corr.index[0]} with correlation {corr.iloc[0]:.3f}.")
    if regression_result:
        metrics = regression_result["metrics"]
        summary.append(
            f"The regression model achieved R-squared {metrics['R-squared']:.3f} and RMSE {metrics['RMSE']:.3f} on the test set."
        )
    return summary


def page_final_summary() -> None:
    title_block("Final Results Summary")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    target = default_target_column(numeric_cols)
    regression_result = st.session_state.get("regression_result")
    summary = build_results_summary(df, metadata, target, regression_result)

    st.markdown("### Key Findings")
    for item in summary:
        st.write(f"- {item}")

    if target:
        values = pd.to_numeric(df[target], errors="coerce")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card(f"Mean {target}", f"{values.mean():.2f}", "Average performance")
        with c2:
            card("Median", f"{values.median():.2f}", "Middle score")
        with c3:
            card("Std Dev", f"{values.std(ddof=1):.2f}", "Spread")
        with c4:
            card("Range", f"{values.max() - values.min():.2f}", "Max minus min")

def create_report_graphs(df: pd.DataFrame, metadata: dict[str, object]) -> list[str]:
    graph_paths = list(st.session_state.get("graph_paths", []))
    numeric_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]
    target = default_target_column(numeric_cols)

    if target:
        graph_paths.append(save_figure(histogram(df, target), f"report_histogram_{target}.png"))
        graph_paths.append(save_figure(box_plot(df, target), f"report_box_plot_{target}.png"))
    if len(numeric_cols) >= 2:
        heatmap_cols = numeric_cols[: min(7, len(numeric_cols))]
        graph_paths.append(save_figure(correlation_heatmap(df, heatmap_cols), "report_correlation_heatmap.png"))
    if categorical_cols:
        graph_paths.append(save_figure(bar_chart(df, categorical_cols[0]), f"report_bar_{categorical_cols[0]}.png"))
    regression_result = st.session_state.get("regression_result")
    if regression_result:
        graph_paths.append(
            save_figure(
                regression_result["actual_predicted_fig"],
                f"report_actual_vs_predicted_{regression_result['target']}.png",
            )
        )

    deduped: list[str] = []
    for path in graph_paths:
        if path not in deduped and Path(path).exists():
            deduped.append(path)
    st.session_state.graph_paths = deduped
    return deduped


def page_report_generator() -> None:
    title_block("Report Generator")
    df, metadata = get_clean_data()
    numeric_cols = metadata["numerical_columns"]
    target = default_target_column(numeric_cols)

    st.markdown("### Report Details")
    team_name = st.text_input("Team name", value=TEAM_NAME)
    dataset_source = st.session_state.dataset_source
    st.write(f"Dataset source: {dataset_source}")
    if hasattr(st, "data_editor"):
        members = st.data_editor(members_placeholder(), num_rows="dynamic", use_container_width=True)
    else:
        members = members_placeholder()
        st.dataframe(members, use_container_width=True, hide_index=True)
        st.caption("Update this placeholder table after opening the generated Word report if your Streamlit version has no data editor.")

    st.markdown("### Optional: Train Default Regression for Report")
    if st.session_state.get("regression_result") is None and target:
        if st.button("Train default regression model for report"):
            try:
                predictors = default_predictors(df, target, metadata["numerical_columns"], metadata["categorical_columns"])
                st.session_state.regression_result = train_regression_model(df, target, predictors)
                st.success("Default regression model added to the report context.")
            except Exception as exc:
                st.error(f"Could not train default model: {exc}")

    if st.button("Generate Word Report", type="primary"):
        try:
            graph_paths = create_report_graphs(df, metadata)
            descriptive = descriptive_statistics(df, numeric_cols[: min(6, len(numeric_cols))]) if numeric_cols else pd.DataFrame()
            regression_result = st.session_state.get("regression_result")
            summary = build_results_summary(df, metadata, target, regression_result)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"student_performance_stats_report_{timestamp}.docx"
            data_description = {
                "Rows": f"{df.shape[0]:,}",
                "Columns": f"{df.shape[1]:,}",
                "Numerical variables": ", ".join(metadata["numerical_columns"][:12]),
                "Categorical variables": ", ".join(metadata["categorical_columns"][:12]),
                "Sample vs population": "Loaded rows are treated as the sample; population means all students represented by the study context.",
            }
            code_files = [
                PROJECT_ROOT / "app.py",
                PROJECT_ROOT / "src" / "data_loader.py",
                PROJECT_ROOT / "src" / "preprocessing.py",
                PROJECT_ROOT / "src" / "eda.py",
                PROJECT_ROOT / "src" / "probability.py",
                PROJECT_ROOT / "src" / "distributions.py",
                PROJECT_ROOT / "src" / "inference.py",
                PROJECT_ROOT / "src" / "regression.py",
                PROJECT_ROOT / "src" / "supplementary_r_syntax.py",
                PROJECT_ROOT / "src" / "report_generator.py",
            ]
            report_path = generate_report(
                output_path=output_path,
                project_title=PROJECT_TITLE,
                team_name=team_name,
                members=members,
                dataset_source=dataset_source,
                data_description=data_description,
                descriptive_stats=descriptive,
                results_summary=summary,
                graph_paths=graph_paths,
                regression_results=regression_result,
                logo_path=LOGO_PATH,
                code_files=code_files,
                r_syntax_references=R_SYNTAX_REFERENCES,
            )
            st.success(f"Report generated: {report_path}")
            st.download_button(
                "Download Word Report",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


def main() -> None:
    initialize_state()
    page = sidebar_navigation()

    if page == "home":
        page_home()
    elif page == "dataset":
        page_dataset_overview()
    elif page == "cleaning":
        page_cleaning()
    elif page == "descriptive":
        page_descriptive_statistics()
    elif page == "eda":
        page_eda_visualizations()
    elif page == "probability":
        page_probability_analysis()
    elif page == "distributions":
        page_distributions()
    elif page == "inference":
        page_inference()
    elif page == "regression":
        page_regression()
    elif page == "summary":
        page_final_summary()
    elif page == "report":
        page_report_generator()


if __name__ == "__main__":
    main()
