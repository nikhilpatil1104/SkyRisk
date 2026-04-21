import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_loader import load_all, get_airline_map
from risk_engine import get_flight_risk, get_alternatives, get_best_time_to_fly
from llm_handler import (
    extract_flight_from_image,
    extract_flight_from_text,
    generate_recommendation,
    chat_response,
    infer_risk_from_similar,
)
from style_loader import inject_styles, inject_theme_toggle

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyRisk — Flight Risk Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
inject_theme_toggle()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return load_all()

dfs         = load_data()
airline_map = get_airline_map(dfs)
time_info   = get_best_time_to_fly(dfs)

ALL_AIRPORTS = sorted(dfs["delay"]["origin"].unique().tolist())
ALL_AIRLINES = sorted(dfs["reliability"]["airline"].tolist())
MONTHS = {
    1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
    7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
}

VERDICT_BADGE = {
    "Go Ahead":              '<span class="badge badge-go">✅ Go Ahead</span>',
    "Book with Caution":     '<span class="badge badge-caution">⚠️ Book with Caution</span>',
    "Consider Alternatives": '<span class="badge badge-consider">🔄 Consider Alternatives</span>',
}
RISK_BADGE = {
    "Low":    '<span class="badge badge-low">🟢 Low Risk</span>',
    "Medium": '<span class="badge badge-medium">🟡 Medium Risk</span>',
    "High":   '<span class="badge badge-high">🔴 High Risk</span>',
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "chat_history": [], "flight_context": None, "risk_result": None,
    "alternatives": [], "llm_rec": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sb-hero">
        <div class="sb-hero-img"></div>
        <div class="sb-hero-content">
            <div class="sb-logo-text">Sky<span>Risk</span></div>
            <div class="sb-logo-sub">Flight Intelligence</div>
        </div>
    </div>
    <div class="sb-section-head">📊 Coverage</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Airports", "20")
    c2.metric("Airlines", len(dfs["reliability"]))
    st.metric("Route-Month Pairs", f"{len(dfs['delay']):,}")
    st.caption("10 years · 2015–2024 · US Domestic")

    st.markdown('<div class="sb-section-head">🕐 Best Times to Fly</div>', unsafe_allow_html=True)
    if time_info["best_hours"]:
        hrs = ", ".join(time_info["best_hours"][:3])
        st.markdown(f'<div class="sb-time-pill"><div class="sb-time-label">✦ Best Hours</div><div class="sb-time-value">{hrs}</div></div>', unsafe_allow_html=True)
    if time_info["best_days"]:
        days = ", ".join(time_info["best_days"][:3])
        st.markdown(f'<div class="sb-time-pill"><div class="sb-time-label">✦ Best Days</div><div class="sb-time-value">{days}</div></div>', unsafe_allow_html=True)

    if st.session_state.flight_context:
        ctx = st.session_state.flight_context
        cat = ctx["risk_category"]
        st.markdown('<div class="sb-section-head">🎯 Active Flight</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card risk-card-{cat.lower()}" style="padding:13px 16px">
            <div style="font-size:1.1rem;font-weight:700;color:var(--text-h);font-family:var(--font-display)">{ctx['origin']} <span style="color:var(--text-muted)">→</span> {ctx['destination']}</div>
            <div style="font-size:0.8rem;color:var(--text-muted);margin:3px 0">{ctx['airline_name']} · {MONTHS[ctx['month']]}</div>
            <div style="margin-top:8px">{RISK_BADGE[cat]}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍  Flight Analysis", "💬  AI Chat", "📊  Explore Data"])


# ── Shared analysis runner ──
def run_analysis(origin, destination, airline_code, month):
    result = get_flight_risk(dfs, origin, destination, airline_code, month)
    if result is None:
        result = infer_risk_from_similar(dfs, origin.upper(), destination.upper(), airline_code.upper(), month)
    alts = get_alternatives(dfs, origin, destination, month, result["risk_score"])
    rec  = generate_recommendation(result, alts, time_info)
    st.session_state.risk_result    = result
    st.session_state.alternatives   = alts
    st.session_state.llm_rec        = rec
    st.session_state.flight_context = result


def render_rec_cards(rec: dict):
    if not isinstance(rec, dict):
        st.info(rec)
        return
    verdict    = rec.get("verdict", "Book with Caution")
    badge_html = VERDICT_BADGE.get(verdict, f"<strong>{verdict}</strong>")
    summary    = rec.get("summary", "")
    inferred   = rec.get("inferred", False)

    st.markdown(f"""
    <div class="risk-card" style="margin-bottom:18px">
        <div>
            <div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;color:var(--text-muted);margin-bottom:9px">✨ AI Verdict</div>
            {badge_html}
            <p style="margin-top:13px;color:var(--text-body);font-size:0.93rem;line-height:1.65">{summary}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if inferred:
        st.markdown('<div class="inferred-notice">ℹ️ Some metrics are statistically inferred from similar routes — exact data for this combination was unavailable.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    best     = rec.get("best_option", {})
    cheapest = rec.get("cheapest_option", {})
    reliable = rec.get("most_reliable", {})

    with c1:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-label label-best">🏆 Best Option</div>
            <h4>{best.get('airline','—')}</h4>
            <p>{best.get('reason','')}</p>
            <div style="margin-top:9px;font-size:0.86rem;color:var(--text-h);font-weight:500">Risk: {best.get('risk_score','—')} &nbsp;·&nbsp; {best.get('fare','—')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-label label-cheap">💰 Cheapest Option</div>
            <h4>{cheapest.get('airline','—')}</h4>
            <p>{cheapest.get('reason','')}</p>
            <div style="margin-top:9px;font-size:0.86rem;color:var(--text-h);font-weight:500">{cheapest.get('fare','—')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-label label-reliable">🛡️ Most Reliable</div>
            <h4>{reliable.get('airline','—')}</h4>
            <p>{reliable.get('reason','')}</p>
            <div style="margin-top:9px;font-size:0.86rem;color:var(--text-h);font-weight:500">Reliability: {reliable.get('reliability','—')}</div>
        </div>
        """, unsafe_allow_html=True)

    fare_adv = rec.get("fare_advice", "")
    time_tip = rec.get("timing_tip", "")
    rf       = rec.get("risk_factors", [])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-label label-cheap">💳 Fare Advice</div>
            <p style="color:var(--text-body);margin-top:7px;font-size:0.91rem">{fare_adv}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-card-label label-best">⏰ Timing Tip</div>
            <p style="color:var(--text-body);margin-top:7px;font-size:0.91rem">{time_tip}</p>
        </div>
        """, unsafe_allow_html=True)

    if rf:
        factors_html = "".join([f'<li style="color:var(--text-body);font-size:0.86rem;margin:5px 0">{f}</li>' for f in rf])
        st.markdown(f"""
        <div class="rec-card" style="margin-top:5px">
            <div class="rec-card-label" style="color:var(--coral)">⚠️ Risk Factors</div>
            <ul style="margin:9px 0 0;padding-left:18px">{factors_html}</ul>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 1 — Flight Analysis
# ════════════════════════════════════════════════════════════
with tab1:
    # ── Hero with real airplane photo ──
    st.markdown("""
    <div class="sky-hero">
        <div class="sky-hero-bg analysis-bg"></div>
        <div class="sky-hero-overlay analysis-ov"></div>
        <div class="sky-hero-content">
            <div class="sky-hero-eyebrow">AI-Powered Flight Intelligence</div>
            <h1 class="sky-hero-title">Fly Smarter.<br><em>Risk Less.</em></h1>
            <p class="sky-hero-sub">
                Analyze delay risk, compare airline reliability, and get personalized booking advice powered by 10 years of US domestic flight data.
            </p>
            <div class="sky-hero-pills">
                <span class="sky-pill">✈ Delay Prediction</span>
                <span class="sky-pill">📊 Airline Ranking</span>
                <span class="sky-pill">💬 AI Chat Advisor</span>
                <span class="sky-pill">📸 Screenshot Analysis</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Destination photo strip ──
    st.markdown("""
    <div class="dest-strip">
        <div class="dest-card">
            <img src="https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400&q=70" alt="NYC">
            <span class="dest-card-label">New York</span>
        </div>
        <div class="dest-card">
            <img src="https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400&q=70" alt="LA">
            <span class="dest-card-label">Los Angeles</span>
        </div>
        <div class="dest-card">
            <img src="https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=400&q=70" alt="Chicago">
            <span class="dest-card-label">Chicago</span>
        </div>
        <div class="dest-card">
            <img src="https://as1.ftcdn.net/v2/jpg/01/31/49/46/1000_F_131494684_pzUEMzSUuzuTItDLkxa6j84kcfyqUzY9.jpg" alt="Miami">
            <span class="dest-card-label">Miami</span>
        </div>
        <div class="dest-card">
            <img src="https://images.unsplash.com/photo-1449034446853-66c86144b0ad?w=400&q=70" alt="San Francisco">
            <span class="dest-card-label">San Francisco</span>
        </div>
        <div class="dest-card">
            <img src="https://images.unsplash.com/photo-1518639192441-8fce0a366e2e?w=400&q=70" alt="Las Vegas">
            <span class="dest-card-label">Las Vegas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form + image upload ──
    col_form, col_img = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="sky-label">✏️ Enter Flight Details</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            origin = st.selectbox(
                "Origin Airport", ALL_AIRPORTS,
                index=ALL_AIRPORTS.index("ORD") if "ORD" in ALL_AIRPORTS else 0,
            )
        with r1c2:
            avail_dest = sorted(dfs["delay"][dfs["delay"]["origin"] == origin]["destination"].unique().tolist())
            destination = st.selectbox("Destination Airport", avail_dest if avail_dest else ALL_AIRPORTS)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            airline_code = st.selectbox(
                "Airline", ALL_AIRLINES,
                format_func=lambda x: f"{x} — {airline_map.get(x, x)}",
            )
        with r2c2:
            month = st.selectbox("Month of Travel", list(MONTHS.keys()), format_func=lambda x: MONTHS[x])

        if st.button("🔍 Analyze Flight Risk", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                run_analysis(origin, destination, airline_code, month)

    with col_img:
        st.markdown('<div class="sky-label">📸 Upload a Screenshot</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-hero-zone">
            <div class="upload-icon-big">🖼️</div>
            <div class="upload-title">Drop your booking screenshot</div>
            <div class="upload-sub">Google Flights, Kayak, Expedia, MakeMyTrip — any site works</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop screenshot here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
            if st.button("📷 Extract & Analyze Screenshot", use_container_width=True):
                with st.spinner("Reading screenshot with GPT-4o Vision..."):
                    image_bytes = uploaded.read()
                    extracted   = extract_flight_from_image(image_bytes)
                if extracted:
                    o = extracted.get("origin")      or origin
                    d = extracted.get("destination") or destination
                    a = extracted.get("airline")     or airline_code
                    m = extracted.get("month")       or month
                    with st.spinner("Analyzing extracted flight..."):
                        run_analysis(o, d, a, m)
                    st.success(f"Extracted: **{o} → {d}** · {airline_map.get(a, a)} · {MONTHS[m]}")
                else:
                    with st.spinner("Using current form values..."):
                        run_analysis(origin, destination, airline_code, month)

    # ── Results ──
    if st.session_state.risk_result:
        r   = st.session_state.risk_result
        cat = r["risk_category"]

        st.markdown("---")

        st.markdown(f"""
        <div class="route-header">
            <h2 style="margin:0;font-family:var(--font-display);font-size:1.9rem;letter-spacing:-0.3px;color:var(--text-h)">{r['origin']} <span style="color:var(--text-muted)">→</span> {r['destination']}</h2>
            <span style="color:var(--text-muted);font-size:0.9rem">{r['airline_name']} ({r['airline']}) &nbsp;·&nbsp; {MONTHS[r['month']]}</span>
            {RISK_BADGE[cat]}
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Risk Score",  f"{r['risk_score']:.2f}")
        m2.metric("Delay Prob",  f"{r['delay_probability']*100:.1f}%")
        m3.metric("Weather",     f"{r['weather_severity']*100:.1f}/100")
        m4.metric("Congestion",  f"{r['congestion_score']*100:.1f}/100")
        m5.metric("Reliability", f"{r['reliability_score']:.2f}")

        if r.get("current_fare"):
            f1, f2, f3 = st.columns(3)
            trend_delta = r["fare_trend"] if r["fare_trend"] != "Stable" else None
            f1.metric("Current Fare",   f"${r['current_fare']:.0f}")
            f2.metric("Predicted Fare", f"${r['predicted_fare']:.0f}", delta=trend_delta)
            f3.metric("Booking Advice", r["fare_recommendation"])

        # Charts — theme-aware colors
        col_gauge, col_bar = st.columns(2)
        color_map = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#fb7185"}
        plot_bg = "rgba(0,0,0,0)"  # transparent so CSS bg-card shows through

        PLOT_LAYOUT = dict(
            paper_bgcolor=plot_bg, plot_bgcolor=plot_bg,
            font={"family": "Outfit, sans-serif", "size": 12, "color": "#8fa3cc"},
            margin=dict(t=45, b=15, l=15, r=15),
            height=270,
        )

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r["risk_score"],
                number={"font": {"size": 38, "family": "Cormorant Garamond, serif"}, "suffix": ""},
                title={"text": "Travel Risk Score", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": "#1e2d55"},
                    "bar":  {"color": color_map[cat], "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 0.33],   "color": "rgba(52,211,153,0.1)"},
                        {"range": [0.33, 0.66], "color": "rgba(251,191,36,0.1)"},
                        {"range": [0.66, 1.0],  "color": "rgba(251,113,133,0.1)"},
                    ],
                    "threshold": {"line": {"color": color_map[cat], "width": 3}, "thickness": 0.8, "value": r["risk_score"]},
                },
            ))
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            components = {
                "Delay (×0.4)":       r["delay_probability"]       * 0.4,
                "Weather (×0.2)":     r["weather_severity"]        * 0.2,
                "Congestion (×0.2)":  r["congestion_score"]        * 0.2,
                "Low Reliability":    (1 - r["reliability_score"]) * 0.2,
            }
            fig2 = px.bar(
                x=list(components.values()), y=list(components.keys()),
                orientation="h", title="Risk Score Breakdown",
                color=list(components.values()),
                color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
                range_color=[0, 0.4],
            )
            fig2.update_layout(
                **PLOT_LAYOUT,
                showlegend=False,
                coloraxis_showscale=False,
                xaxis={"gridcolor": "rgba(100,140,255,0.1)", "zerolinecolor": "rgba(100,140,255,0.1)"},
                yaxis={"gridcolor": "rgba(100,140,255,0.1)"},
                title_font={"size": 13},
            )
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sky-label">🤖 AI Recommendation</div>', unsafe_allow_html=True)
        render_rec_cards(st.session_state.llm_rec)

        if st.session_state.alternatives:
            st.markdown('<div class="sky-label">✅ Better Alternatives on This Route</div>', unsafe_allow_html=True)
            for i, alt in enumerate(st.session_state.alternatives):
                cat_alt  = alt["risk_category"]
                fare_str = f"${alt['predicted_fare']:.0f}" if alt["predicted_fare"] else "N/A"
                rank_icon = ["🥇","🥈","🥉"][i] if i < 3 else "•"
                st.markdown(f"""
                <div class="alt-row">
                    <div style="display:flex;align-items:center;gap:13px">
                        <span style="font-size:1.25rem">{rank_icon}</span>
                        <div>
                            <div style="font-weight:600;color:var(--text-h)">{alt['airline_name']} <span style="color:var(--text-muted);font-size:0.82rem">({alt['airline']})</span></div>
                            <div style="font-size:0.8rem;color:var(--text-muted)">Delay: {alt['delay_probability']*100:.1f}% · Reliability: {alt['reliability_score']:.2f}</div>
                        </div>
                    </div>
                    <div style="text-align:right">
                        {RISK_BADGE[cat_alt]}
                        <div style="font-size:0.86rem;color:var(--text-h);margin-top:5px;font-weight:500">Risk: {alt['risk_score']} · {fare_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif r["risk_category"] == "Low":
            st.markdown('<div class="rec-card" style="border-left:3.5px solid var(--emerald)"><p style="color:var(--emerald);margin:0;font-weight:500">✅ Your selected airline is already the lowest-risk option on this route.</p></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — AI Chat
# ════════════════════════════════════════════════════════════
with tab2:
    # ── Hero ──
    st.markdown("""
    <div class="sky-hero" style="min-height:200px">
        <div class="sky-hero-bg chat-bg"></div>
        <div class="sky-hero-overlay chat-ov"></div>
        <div class="sky-hero-content" style="padding:2rem 3rem">
            <div class="sky-hero-eyebrow">Your Personal Flight Advisor</div>
            <h2 class="sky-hero-title" style="font-size:2.4rem">Ask Me <em>Anything</em> ✈️</h2>
            <p class="sky-hero-sub" style="font-size:0.9rem">
                Ask about delays, best travel times, fare trends, airline reliability — I'm here to help you travel smarter.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ctx = st.session_state.flight_context
    if ctx:
        cat = ctx["risk_category"]
        st.markdown(f"""
        <div class="chat-ctx-banner ctx-{cat.lower()}">
            <div>
                <div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:1.8px;color:var(--text-muted);margin-bottom:5px">Currently Analyzing</div>
                <div style="font-weight:700;color:var(--text-h);font-size:1.08rem;font-family:var(--font-display)">{ctx['origin']} <span style="color:var(--text-muted)">→</span> {ctx['destination']}</div>
                <div style="font-size:0.82rem;color:var(--text-body);margin-top:2px">{ctx['airline_name']} &nbsp;·&nbsp; {MONTHS[ctx['month']]}</div>
            </div>
            <div>{RISK_BADGE[cat]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:18px">
            <p style="color:var(--text-muted);font-size:0.88rem;margin:0">
                💡 Analyze a flight in <strong style="color:var(--accent)">Flight Analysis</strong> first for full context, or ask any general flight question below!
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Screenshot upload for chat — premium card ──
    st.markdown('<div class="sky-label" style="margin-top:24px">📎 Upload a Screenshot to Analyze</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chat-attach-zone">
        <span class="chat-attach-icon">🖼️</span>
        <span class="chat-attach-text">Drop a flight screenshot — <strong>I'll read it instantly</strong> and give you a full risk breakdown</span>
    </div>
    """, unsafe_allow_html=True)
    chat_img = st.file_uploader(
        "Upload flight screenshot",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
        key="chat_image"
    )

    if chat_img and chat_img != st.session_state.get("_last_chat_img"):
        st.session_state["_last_chat_img"] = chat_img
        with st.spinner("Reading your screenshot... ✨"):
            extracted = extract_flight_from_image(chat_img.read())
        if extracted and (extracted.get("origin") or extracted.get("destination")):
            o = extracted.get("origin")      or (ctx["origin"] if ctx else "ORD")
            d = extracted.get("destination") or (ctx["destination"] if ctx else "JFK")
            a = extracted.get("airline")     or (ctx["airline"] if ctx else "DL")
            m = extracted.get("month")       or (ctx["month"] if ctx else 6)
            result = get_flight_risk(dfs, o, d, a, m) or infer_risk_from_similar(dfs, o, d, a, m)
            st.session_state.flight_context = result
            auto_msg = f"I uploaded a screenshot — looks like **{o} → {d}** on **{airline_map.get(a,a)}** in **{MONTHS[m]}**. What can you tell me? 🧐"
            st.session_state.chat_history.append({"role": "user", "content": auto_msg})
            reply = chat_response(st.session_state.chat_history, st.session_state.flight_context)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about delays, fares, best times, airline reliability... ✈️")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(""):
                extracted = extract_flight_from_text(user_input)
                if extracted and extracted.get("origin") and extracted.get("destination"):
                    o = extracted["origin"]
                    d = extracted["destination"]
                    a = extracted.get("airline") or "DL"
                    m = extracted.get("month")   or 6
                    result = get_flight_risk(dfs, o, d, a, m) or infer_risk_from_similar(dfs, o, d, a, m)
                    if result:
                        st.session_state.flight_context = result
                reply = chat_response(
                    st.session_state.chat_history,
                    flight_context=st.session_state.flight_context,
                )
                st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 3 — Explore Data
# ════════════════════════════════════════════════════════════
with tab3:
    # ── Hero ──
    st.markdown("""
    <div class="sky-hero" style="min-height:190px">
        <div class="sky-hero-bg explore-bg"></div>
        <div class="sky-hero-overlay explore-ov"></div>
        <div class="sky-hero-content" style="padding:2rem 3rem">
            <div class="sky-hero-eyebrow">10 Years of US Domestic Data</div>
            <h2 class="sky-hero-title" style="font-size:2.4rem">Explore the <em>Numbers</em> 📊</h2>
            <p class="sky-hero-sub" style="font-size:0.88rem">
                Dig into airline reliability scores, airport congestion maps, weather patterns, and optimal travel times.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    explore_tab = st.selectbox(
        "Dataset",
        ["Airline Reliability", "Airport Congestion", "Weather Severity by Month", "Time Recommendations"],
        label_visibility="collapsed",
    )

    PLOT_BG = "rgba(0,0,0,0)"
    PLOT_LAYOUT = dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font={"color": "#8fa3cc", "family": "Outfit, sans-serif", "size": 12},
        xaxis={"gridcolor": "rgba(100,140,255,0.1)", "zerolinecolor": "rgba(100,140,255,0.08)"},
        yaxis={"gridcolor": "rgba(100,140,255,0.1)", "zerolinecolor": "rgba(100,140,255,0.08)"},
        height=400,
        margin=dict(t=50, b=20, l=10, r=10),
    )

    if explore_tab == "Airline Reliability":
        df = dfs["reliability"][["airline","airline_name","delay_rate_pct","reliability_score","reliability_rank"]].copy()
        df.columns = ["Code","Airline","Delay Rate %","Reliability Score","Rank"]
        df = df.sort_values("Rank")
        fig = px.bar(
            df, x="Airline", y="Reliability Score",
            color="Reliability Score", color_continuous_scale="RdYlGn",
            range_color=[0,1], title="Airline Reliability Scores (higher = better)",
            text="Reliability Score",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif explore_tab == "Airport Congestion":
        df = dfs["congestion"].sort_values("congestion_score", ascending=False)
        fig = px.bar(
            df, x="airport", y="congestion_score",
            color="congestion_score", color_continuous_scale="RdYlGn_r",
            title="Airport Congestion Scores (higher = more congested)",
            text="congestion_score",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif explore_tab == "Weather Severity by Month":
        df    = dfs["weather"].copy()
        pivot = df.pivot(index="airport", columns="month", values="weather_severity")
        fig   = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            title="Weather Severity Heatmap — Airport × Month",
            labels={"x":"Month","y":"Airport","color":"Severity"},
            aspect="auto",
        )
        fig.update_xaxes(tickvals=list(range(1,13)), ticktext=[MONTHS[m][:3] for m in range(1,13)])
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif explore_tab == "Time Recommendations":
        dim_labels = {"hour":"By Hour","day_of_week":"By Day","month":"By Month","season":"By Season"}
        for dim, label in dim_labels.items():
            sub = dfs["time_rec"][dfs["time_rec"]["dimension"] == dim][
                ["label","delay_rate_pct","cancellation_rate","recommendation"]
            ].copy()
            if not sub.empty:
                st.markdown(f'<div class="sky-label">{label}</div>', unsafe_allow_html=True)
                st.dataframe(
                    sub.rename(columns={"label":"Time","delay_rate_pct":"Delay %","cancellation_rate":"Cancel %","recommendation":"Verdict"}),
                    use_container_width=True, hide_index=True,
                )
