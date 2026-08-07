from pathlib import Path

import streamlit as st


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "custom.css"

    if not css_path.exists():
        st.warning(f"CSS file not found: {css_path}")
        return

    css_content = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )