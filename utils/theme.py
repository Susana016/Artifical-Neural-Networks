import streamlit as st

def apply_theme(primary: str, background: str, text: str, secondary: str = None):
    if secondary is None:
        secondary = primary  # fallback to primary if not provided
    # Derive lighter tints from the primary for backgrounds/borders
    st.markdown(f"""
    <style>
        /* Top toolbar bar */
        header[data-testid="stHeader"] {{
            background-color: {secondary} !important;
        }}

        /* Top bar icons (hamburger, deploy, etc.) */
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg {{
            color: white !important;
            fill: white !important;
        }}

        /* Fullscreen / expand image button */
        button[data-testid="StyledFullScreenButton"],
        button[title="View fullscreen"],
        div[data-testid="stImage"] button,
        div[data-testid="stPlotlyChart"] button,
        div[data-testid="element-container"] button[kind="icon"] {{
            background-color: {primary} !important;
            color: white !important;
            border: none !important;
        }}
        button[data-testid="StyledFullScreenButton"] svg,
        button[title="View fullscreen"] svg,
        div[data-testid="stImage"] button svg,
        div[data-testid="stPlotlyChart"] button svg,
        div[data-testid="element-container"] button[kind="icon"] svg {{
            fill: white !important;
            stroke: white !important;
        }}
        .stApp {{ background-color: {background}; }}

        /* Headings */
        h1, h2, h3, h4 {{ color: {primary}; }}

        /* Body text */
        p, li, span {{ color: {text}; }}

        /* Labels on inputs */
        label {{ color: {text} !important; font-weight: 600; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{ background-color: {background}; }}

        /* All buttons — force white text always */
        div.stButton > button, 
        div.stButton > button p {{
            color: white !important;
        }}

        /* Primary buttons */
        div.stButton > button[kind="primary"] {{
            background: {primary};
            color: white !important;
            border: none;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: {secondary};
            color: white !important;
        }}

        /* Secondary buttons */
        div.stButton > button[kind="secondary"] {{
            background: transparent;
            border: 2px solid {secondary};
            color: {secondary} !important;
        }}
        div.stButton > button[kind="secondary"]:hover {{
            background: {secondary};
            color: white !important;
        }}

        /* Sliders */
        .stSlider > div > div > div > div {{ background: {primary} !important; }}

        /* Selectbox — box fill + border */
        div[data-baseweb="select"] > div {{
            background-color: {secondary} !important;
            border-color: {secondary} !important;
        }}
        div[data-baseweb="select"] > div:hover {{
            border-color: {secondary} !important;
        }}

        /* Selectbox text white so it's readable on colored bg */
        div[data-baseweb="select"] span {{
            color: white !important;
        }}

        /* Dropdown arrow white */
        div[data-baseweb="select"] svg {{
            fill: white !important;
        }}

        /* Selected option in dropdown */
        li[aria-selected="true"] {{
            background-color: {secondary} !important;
            color: white !important;
        }}

        /* Expander header background */
        details > summary,
        div[data-testid="stExpander"] > details > summary {{
            color: white !important;
            font-weight: 700 !important;
            background-color: {secondary} !important;
        }}

        /* Expander header icon/arrow */
        div[data-testid="stExpander"] > details > summary svg {{
            fill: white !important;
        }}

        /* Expander body background */
        details > div {{
            background-color: {background} !important;
        }}

        /* Expander border */
        details {{
            border-color: {secondary} !important;
        }}

        /* Selectbox text white so it's readable on colored bg */
        div[data-baseweb="select"] span {{
            color: white !important;
        }}

        /* Tab text */
        div[data-baseweb="tab"] button {{
            color: {text} !important;
        }}

        /* Active tab underline */
        div[data-baseweb="tab-highlight"] {{
            background-color: {primary} !important;
        }}

        /* Table header */
        thead tr th {{
            color: {primary} !important;
            border-bottom: 2px solid {secondary} !important;
        }}

        /* Table body cells */
        tbody tr td {{
            color: {text} !important;
        }}

        /* Table row hover */
        tbody tr:hover td {{
            background-color: {secondary}22 !important;
        }}

        /* Inline code */
        code {{
            color: {secondary} !important;
            background: {background} !important;
        }}
        div[data-testid="metric-container"] {{
            border-left: 4px solid {primary};
        }}

        /* Divider */
        hr {{ border-color: {primary}44; }}

        /* Model cards */
        .model-card {{
            border: 2px solid {primary}44;
            box-shadow: 0 4px 16px {primary}22;
        }}
        .model-card:hover {{
            box-shadow: 0 8px 24px {primary}44;
        }}
        .card-title  {{ color: {primary}; }}
        .card-badge  {{ background: {background}; color: {text}; }}
        .stat-box    {{ background: {background}; color: {text}; }}
    </style>
    """, unsafe_allow_html=True)