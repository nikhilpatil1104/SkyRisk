import base64
import json
import os
import re
from openai import OpenAI

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY  = os.environ.get("TAVILY_API_KEY", "")   # set this in your env

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_VISION = "gpt-4o"
MODEL_CHAT   = "gpt-4o-mini"

# ── Tavily client (lazy-loaded so app still runs if key is missing) ──
_tavily = None
def _get_tavily():
    global _tavily
    if _tavily is None and TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            _tavily = TavilyClient(api_key=TAVILY_API_KEY)
        except ImportError:
            print("[tavily] Package not installed — run: pip install tavily-python")
    return _tavily


# ── Keywords that signal a real-time / live-data query ──
_REALTIME_SIGNALS = [
    "price", "fare", "cost", "cheap", "cheapest", "how much",
    "book", "buy", "ticket", "deal", "sale", "right now", "today",
    "current", "live", "latest", "2025", "2026", "2027",
    "flight number", "schedule", "depart", "arrive", "duration",
    "layover", "direct", "nonstop", "status", "delay today",
    "weather forecast", "visa", "passport", "entry requirement",
    "airline news", "strike", "cancel",
]

def _needs_web_search(user_message: str) -> bool:
    """Return True if the latest user message likely needs live data."""
    msg = user_message.lower()
    return any(kw in msg for kw in _REALTIME_SIGNALS)


def _tavily_search(query: str, max_results: int = 4) -> str:
    """
    Run a Tavily search and return a compact context string to inject
    into the system prompt. Returns empty string on any failure.
    """
    tv = _get_tavily()
    if not tv:
        return ""
    try:
        resp = tv.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,          # Tavily's own AI summary
        )
        parts = []

        # Tavily's concise AI answer (most useful)
        if resp.get("answer"):
            parts.append(f"SEARCH SUMMARY: {resp['answer']}")

        # Individual results for more context
        for r in resp.get("results", [])[:max_results]:
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = r.get("content", "")[:300]   # trim long snippets
            parts.append(f"• {title} ({url})\n  {content}")

        return "\n\n".join(parts)
    except Exception as e:
        print(f"[tavily error] {e}")
        return ""


def _clean_json(raw: str) -> str:
    return re.sub(r"```json|```", "", raw).strip()


def _safe_json(raw: str) -> dict | None:
    try:
        return json.loads(_clean_json(raw))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Image / text extractors (unchanged)
# ─────────────────────────────────────────────────────────────────

def extract_flight_from_image(image_bytes: bytes) -> dict | None:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = """You are a flight data extractor. Read this flight booking screenshot carefully.
Return ONLY a valid JSON object with these exact keys:
{
  "origin": "3-letter IATA airport code — infer from city name if needed (Chicago=ORD, New York=JFK, LA=LAX)",
  "destination": "3-letter IATA airport code",
  "airline": "2-letter IATA airline code (DL=Delta, AA=American, UA=United, WN=Southwest, B6=JetBlue, NK=Spirit, F9=Frontier, AS=Alaska)",
  "month": integer 1-12,
  "raw_airline_name": "airline name as it appears in the image",
  "raw_origin": "origin city/airport as shown",
  "raw_destination": "destination city/airport as shown"
}
Always make your best inference. Never return null. Return only the JSON object, no explanation."""
    try:
        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=300, temperature=0.1,
        )
        return _safe_json(response.choices[0].message.content)
    except Exception as e:
        print(f"[image_parser error] {e}")
        return None


def extract_flight_from_text(user_message: str) -> dict | None:
    prompt = f"""You are a flight data extractor. Read the user message and extract flight details.
Return ONLY a valid JSON object:
{{
  "origin": "3-letter IATA code (Chicago=ORD, NYC=JFK, LA=LAX, SF=SFO, Miami=MIA, Dallas=DFW, Boston=BOS, Seattle=SEA, Denver=DEN, Atlanta=ATL, Phoenix=PHX, Las Vegas=LAS, Baltimore=BWI, Newark=EWR) or null",
  "destination": "3-letter IATA code or null",
  "airline": "2-letter IATA code (DL=Delta, AA=American, UA=United, WN=Southwest, B6=JetBlue, NK=Spirit, F9=Frontier, AS=Alaska, HA=Hawaiian) or null",
  "month": integer 1-12 or null
}}
User message: "{user_message}"
Return only the JSON, no explanation."""
    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, temperature=0.1,
        )
        return _safe_json(response.choices[0].message.content)
    except Exception as e:
        print(f"[text_parser error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Risk inference (unchanged)
# ─────────────────────────────────────────────────────────────────

def infer_risk_from_similar(dfs, origin: str, destination: str, airline: str, month: int) -> dict:
    import numpy as np
    from risk_engine import compute_risk_score, risk_category

    delay_df  = dfs["delay"]
    weather_df = dfs["weather"]
    cong_df   = dfs["congestion"]
    rel_df    = dfs["reliability"]

    route_rows = delay_df[
        (delay_df["origin"] == origin) &
        (delay_df["destination"] == destination) &
        (delay_df["month"] == month)
    ]
    if not route_rows.empty:
        delay_prob  = float(route_rows["delay_probability"].mean())
        infer_note  = f"Based on {len(route_rows)} airlines on {origin}→{destination} in month {month}"
    else:
        origin_rows = delay_df[(delay_df["origin"] == origin) & (delay_df["month"] == month)]
        if not origin_rows.empty:
            delay_prob  = float(origin_rows["delay_probability"].mean())
            infer_note  = f"Based on all routes from {origin} in month {month}"
        else:
            airline_rows = delay_df[delay_df["airline"] == airline]
            delay_prob   = float(airline_rows["delay_probability"].mean()) if not airline_rows.empty else 0.25
            infer_note   = f"Based on {airline} historical average"

    w_row        = weather_df[(weather_df["airport"] == origin) & (weather_df["month"] == month)]
    weather_sev  = float(w_row["weather_severity"].mean()) if not w_row.empty else float(weather_df["weather_severity"].mean())

    c_row        = cong_df[cong_df["airport"] == origin]
    cong_score   = float(c_row["congestion_score"].values[0]) if not c_row.empty else 0.5

    r_row = rel_df[rel_df["airline"] == airline]
    if not r_row.empty:
        rel_score    = float(r_row["reliability_score"].values[0])
        airline_name = r_row["airline_name"].values[0]
    else:
        rel_score    = 0.45
        airline_name = airline

    fare_df = dfs["fare"]
    f_row   = fare_df[(fare_df["origin"] == origin) & (fare_df["destination"] == destination)]
    if not f_row.empty:
        current_fare   = float(f_row["current_fare"].values[0])
        predicted_fare = float(f_row["predicted_fare"].values[0])
        fare_trend     = f_row["trend"].values[0]
        fare_rec       = f_row["recommendation"].values[0]
    else:
        current_fare   = float(fare_df["current_fare"].mean())
        predicted_fare = float(fare_df["predicted_fare"].mean())
        fare_trend     = "Stable"
        fare_rec       = "Book when ready"

    score    = compute_risk_score(delay_prob, weather_sev, cong_score, rel_score)
    category = risk_category(score)

    return {
        "origin": origin, "destination": destination, "airline": airline,
        "airline_name": airline_name, "month": month,
        "delay_probability": round(delay_prob, 4),
        "weather_severity":  round(weather_sev, 4),
        "congestion_score":  round(cong_score, 4),
        "reliability_score": round(rel_score, 4),
        "risk_score": score, "risk_category": category,
        "current_fare":   round(current_fare, 2),
        "predicted_fare": round(predicted_fare, 2),
        "fare_trend": fare_trend, "fare_recommendation": fare_rec,
        "inferred": True, "infer_note": infer_note,
    }


# ─────────────────────────────────────────────────────────────────
# Structured recommendation (unchanged)
# ─────────────────────────────────────────────────────────────────

def generate_recommendation(risk_result: dict, alternatives: list, time_info: dict) -> dict:
    alt_text = ""
    if alternatives:
        alt_lines = []
        for a in alternatives:
            fare_str = f"${a['predicted_fare']:.0f}" if a["predicted_fare"] else "N/A"
            alt_lines.append(
                f"- {a['airline_name']} ({a['airline']}): risk {a['risk_score']} ({a['risk_category']}), "
                f"delay {a['delay_probability']*100:.1f}%, reliability {a['reliability_score']:.2f}, est. fare {fare_str}"
            )
        alt_text = "Alternative airlines on this route:\n" + "\n".join(alt_lines)

    best_hours = ", ".join(time_info.get("best_hours", [])[:3]) or "early morning"
    best_days  = ", ".join(time_info.get("best_days",  [])[:3]) or "Tuesday, Wednesday"

    delay_pct   = risk_result["delay_probability"] * 100
    weather_pct = risk_result["weather_severity"]  * 100
    cong_pct    = risk_result["congestion_score"]  * 100
    rel         = risk_result["reliability_score"]
    fare_str    = f"${risk_result['current_fare']:.0f}"   if risk_result.get("current_fare")   else "N/A"
    pred_fare_str = f"${risk_result['predicted_fare']:.0f}" if risk_result.get("predicted_fare") else "N/A"
    inferred_note = f"\nNote: {risk_result.get('infer_note','')}" if risk_result.get("inferred") else ""

    context = f"""Flight analysis data (explain all numbers in plain English):
- Route: {risk_result['origin']} → {risk_result['destination']}
- Airline: {risk_result['airline_name']} ({risk_result['airline']})
- Month: {risk_result['month']}
- Delay Probability: {delay_pct:.1f}% (meaning {int(delay_pct)} out of 100 flights on this route are delayed)
- Weather Severity at origin: {weather_pct:.1f}/100 ({'mild conditions' if weather_pct < 30 else 'moderate weather impact' if weather_pct < 60 else 'severe weather risk'})
- Airport Congestion: {cong_pct:.1f}/100 ({'quiet airport' if cong_pct < 30 else 'busy airport' if cong_pct < 70 else 'very congested — expect delays and long queues'})
- Airline Reliability: {rel:.2f}/1.0 ({'excellent punctuality' if rel > 0.7 else 'average on-time performance' if rel > 0.5 else 'below average — consider alternatives'})
- Overall Risk Score: {risk_result['risk_score']} ({risk_result['risk_category']} Risk)
- Current Fare: {fare_str} → Predicted: {pred_fare_str} | Trend: {risk_result['fare_trend']}
- Fare Action: {risk_result['fare_recommendation']}
{inferred_note}
{alt_text}
Historically best departure hours: {best_hours}
Best days to fly: {best_days}"""

    system_prompt = """You are SkyRisk, a precise AI flight risk advisor.
Analyze the flight data and return ONLY a valid JSON object. Explain ALL numbers in plain English — never show raw scores without meaning.
Return this exact structure:
{
  "verdict": "Go Ahead" | "Book with Caution" | "Consider Alternatives",
  "summary": "2-3 sentence plain-English summary. Explain what delay probability means in everyday terms (e.g. '29% means roughly 1 in 3 flights is delayed'). No raw numbers without explanation.",
  "best_option": {"airline": "full name (code)", "reason": "why best in plain English", "risk_score": 0.0, "fare": "$000"},
  "cheapest_option": {"airline": "full name (code)", "reason": "price advantage and any tradeoff", "fare": "$000"},
  "most_reliable": {"airline": "full name (code)", "reason": "on-time performance in plain English", "reliability": 0.00},
  "fare_advice": "Plain English booking advice — should they book now or wait and why",
  "timing_tip": "Best specific day and time of day to depart and why",
  "risk_factors": ["Plain English factor 1 with explanation", "Plain English factor 2", "Plain English factor 3"]
}
Return ONLY the JSON object, no markdown fences."""

    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ],
            max_tokens=700, temperature=0.3,
        )
        raw    = response.choices[0].message.content.strip()
        parsed = _safe_json(raw)
        if parsed:
            parsed["inferred"] = risk_result.get("inferred", False)
            return parsed
    except Exception as e:
        print(f"[recommendation error] {e}")

    return {
        "verdict": "Book with Caution" if risk_result["risk_score"] > 0.4 else "Go Ahead",
        "summary": f"{risk_result['airline_name']} on {risk_result['origin']}→{risk_result['destination']} shows a {risk_result['risk_category'].lower()} overall risk. About {delay_pct:.0f} out of 100 flights on this route experience delays.",
        "best_option":    {"airline": risk_result["airline_name"], "reason": "Selected airline", "risk_score": risk_result["risk_score"], "fare": fare_str},
        "cheapest_option":{"airline": risk_result["airline_name"], "reason": "Current selection", "fare": fare_str},
        "most_reliable":  {"airline": risk_result["airline_name"], "reason": f"On-time performance: {'good' if rel > 0.6 else 'average'}", "reliability": rel},
        "fare_advice":    f"Fares are {risk_result['fare_trend'].lower()}. {risk_result['fare_recommendation']}.",
        "timing_tip":     f"Fly on {best_days} during {best_hours} for fewest delays historically.",
        "risk_factors": [
            f"Delay risk: {delay_pct:.0f}% of flights delayed — {'acceptable' if delay_pct < 30 else 'worth considering alternatives'}",
            f"Weather at {risk_result['origin']}: {'mild, unlikely to affect your flight' if weather_pct < 30 else 'moderate impact possible' if weather_pct < 60 else 'significant weather risk'}",
            f"Airport congestion: {'light traffic' if cong_pct < 30 else 'busy — arrive early' if cong_pct < 70 else 'very busy — arrive 2+ hours early'}",
        ],
        "inferred": risk_result.get("inferred", False),
    }


# ─────────────────────────────────────────────────────────────────
# Chat response — with Tavily real-time search
# ─────────────────────────────────────────────────────────────────

def chat_response(messages: list, flight_context: dict | None = None) -> str:
    """
    Full conversation. Uses web_search_preview for real-time queries.
    Never refuses — handles future dates, live prices, specific schedules.
    """
    


    # ── Core personality + rules ──
    system_content = """You are Sky, SkyRisk's friendly AI flight advisor ✈️ — like a knowledgeable travel buddy who genuinely cares about helping people fly smarter.
        You have access to:
        1. Historical flight database (2015-2024): delay patterns, airline reliability, airport congestion, weather, fare trends for 20 major US airports
        2. Web search: use this for specific future dates, live prices, current airline schedules

        ABSOLUTE RULES — never break these:
        - NEVER say "I don't have real-time data", "my training cutoff", "I cannot access", or any version of refusal
        - For specific future dates: ALWAYS use web search to find flights, then add reliability context from historical data
        - For "cheapest flight" queries: search the web and return ranked options with prices
        - ALL numbers must be explained in plain English — never show a score without saying what it means
        - Format responses beautifully with emoji headers, bold highlights, and bullet points
        PERSONALITY:
        - Warm, upbeat, direct — like a friend who knows everything about flights
        - Use emojis naturally: ✈️ 🛫 🌤️ ⚡ 💡 ✅ ⚠️ 🔥 💰 🏆 😊 🙌
        - Open with a brief friendly line ("Nice route!", "Great question!", "Oh let me dig into that!")
        - Give confident answers — never hedge, never refuse, never say you can't access data
        - Celebrate good news, be honest about risk without being scary
        - End with one useful follow-up offer

        ANSWER SHAPE — match the question:
        - Price / fare question → give price ranges, cheapest options first. That's the whole answer.
        - Reliability question → one warm paragraph with the actual number.
        - "Should I book now?" → direct yes/no with the reason.
        - General travel question → conversational answer, no template.
        - "Full analysis" / "tell me everything" → then use a structured format.
        - Screenshot / route uploaded → warm 2-3 sentence summary + one clear call to action.

        STRICT RULES:
        - NEVER start with "For your flight from X to Y..." — that's robotic
        - NEVER say you lack real-time data, can't access current prices, or can't help
        - NEVER dump all data fields as bullet points by default
        - NEVER refuse a question — always give your best answer using available data + web search results
        - Numbers need human context: "23% — roughly 1 in 4 flights run late", not just "0.23"
        - If web search results are provided below, USE THEM — cite specific prices, airlines, dates found
        - Keep responses concise — short paragraphs win over long lists"""

    # ── Attach SkyRisk flight context ──
    if flight_context:
        ctx         = flight_context
        delay_pct   = ctx.get("delay_probability", 0) * 100
        weather_pct = ctx.get("weather_severity",  0) * 100
        cong_pct    = ctx.get("congestion_score",  0) * 100
        rel         = ctx.get("reliability_score", 0)
        weather_label = "mild 🌤️"    if weather_pct < 30 else "moderate ⛅" if weather_pct < 60 else "rough 🌩️"
        airport_label = "quiet 😌"   if cong_pct   < 30 else "busy 🚶"     if cong_pct   < 70 else "very congested 😬"
        rel_label     = "excellent ✅" if rel        > 0.7 else "average 🟡"  if rel        > 0.5 else "below average ⚠️"

        system_content += f"""

SKYRISK DATABASE — current flight context (weave naturally into answer, don't dump all at once):
Route: {ctx.get('origin')} → {ctx.get('destination')} | {ctx.get('airline_name')} | Month {ctx.get('month')}
Risk: {ctx.get('risk_category')} (score {ctx.get('risk_score', 0):.2f}/1.0)
Delays: {delay_pct:.0f}% of flights delayed — about {int(delay_pct)} in 100 flights
Weather at origin: {weather_label} ({weather_pct:.0f}/100)
Airport congestion: {airport_label} ({cong_pct:.0f}/100)
Airline reliability: {rel_label} ({rel:.2f}/1.0)
Fare: ${ctx.get('current_fare', 0):.0f} now → ${ctx.get('predicted_fare', 0):.0f} predicted | Trend: {ctx.get('fare_trend', 'Stable')} | Advice: {ctx.get('fare_recommendation', 'Book when ready')}"""

    # ── Run Tavily search if the latest message needs live data ──
    latest_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_msg = m.get("content", "")
            break

    web_context = ""
    if latest_user_msg and _needs_web_search(latest_user_msg):
        # Build a focused search query
        search_query = latest_user_msg
        if flight_context:
            o = flight_context.get("origin", "")
            d = flight_context.get("destination", "")
            if o and d and o not in search_query and d not in search_query:
                search_query = f"{o} to {d} flight {search_query}"
        web_context = _tavily_search(search_query)

    if web_context:
        system_content += f"""

LIVE WEB SEARCH RESULTS (use these to answer with real, current data — cite specific numbers you find):
{web_context}"""

    full_messages = [{"role": "system", "content": system_content}] + messages

    # ── Primary: gpt-4o ──
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=full_messages,
            max_tokens=700,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception as e:
        print(f"[chat gpt-4o error] {e}")

    # ── Fallback: gpt-4o-mini ──
    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=full_messages,
            max_tokens=700,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[chat fallback error] {e}")
        return "Having a brief connection issue ✈️ — give it a second and try again!"
