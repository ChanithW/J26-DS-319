"""Streamlit front end for Component 2. Run from the sub02 folder:

    streamlit run streamlit_app/app.py

It talks only to the Component 2 API (COMPONENT2_API_URL, default http://localhost:8000).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so views can `import helpers`, `import ui`

import streamlit as st

st.set_page_config(page_title="Flare CME predictor", page_icon="🔭", layout="wide")

pages = [
    st.Page("views/predict.py", title="Predict", icon="🔭", default=True),
    st.Page("views/results.py", title="Results", icon="📊"),
    st.Page("views/runs.py", title="Saved runs", icon="🗂️"),
    st.Page("views/model_info.py", title="Model info", icon="ℹ️"),
]
st.navigation(pages).run()
