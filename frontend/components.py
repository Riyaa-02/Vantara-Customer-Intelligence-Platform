import streamlit as st


def metric_card(title, value, delta=None):
    """Render a dashboard metric card."""

    delta_html = ""

    if delta:
        delta_html = (
            f"<div style='color:#16a34a;"
            f"font-size:14px;"
            f"margin-top:6px;'>"
            f"{delta}"
            f"</div>"
        )

    card_html = (
        "<div class='metric-card'>"
        f"<div style='font-size:15px;"
        f"color:#64748B;'>"
        f"{title}"
        "</div>"
        f"<div style='font-size:34px;"
        f"font-weight:700;"
        f"color:#1E293B;"
        f"margin-top:8px;'>"
        f"{value}"
        "</div>"
        f"{delta_html}"
        "</div>"
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


def page_title(title, subtitle=""):
    st.markdown(
        f"""
        <h1 style="margin-bottom:0px;">{title}</h1>
        <p style="color:#64748B;margin-top:4px;">
            {subtitle}
        </p>
        """,
        unsafe_allow_html=True,
    )