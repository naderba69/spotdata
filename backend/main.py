# =============================================================================
# SURFCASTING ANALYTICS ENGINE — FastAPI Backend
# Deployment Target: Render (Free Tier)
# AI Engine: Google Gemini API
# Data Source: Open-Meteo (Weather + Marine APIs)
# =============================================================================

import os
import math
import statistics
import httpx
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Surfcasting Analytics API",
    description="Physics-grade marine analytics engine with Gemini AI synthesis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Gemini Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Request Schema (Pydantic)
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    latitude: float
    longitude: float
    beach_orientation: int          # 0–360 degrees
    beach_type: str                 # 'sandy' or 'rocky'
    target_date: str                # 'today', 'tomorrow', 'day_after'

    @validator("beach_type")
    def validate_beach_type(cls, v):
        if v not in ("sandy", "rocky"):
            raise ValueError("beach_type must be 'sandy' or 'rocky'")
        return v

    @validator("target_date")
    def validate_target_date(cls, v):
        if v not in ("today", "tomorrow", "day_after"):
            raise ValueError("target_date must be 'today', 'tomorrow', or 'day_after'")
        return v

    @validator("beach_orientation")
    def validate_orientation(cls, v):
        if not (0 <= v <= 360):
            raise ValueError("beach_orientation must be between 0 and 360")
        return v


# ===========================================================================
# SECTION 1 — DATA FETCHING (Open-Meteo)
# ===========================================================================

MARINE_API = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

MARINE_VARS = (
    "wave_height,wave_period,"
    "swell_wave_height,swell_wave_period,swell_wave_direction,"
    "wind_wave_height,wind_wave_period"
)
WEATHER_VARS = (
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "pressure_msl,temperature_2m"
)

def _resolve_forecast_days(target_date: str) -> int:
    """Map target_date label to required forecast_days for Open-Meteo."""
    return {"today": 1, "tomorrow": 2, "day_after": 3}[target_date]


def _safe(value, fallback: float = 0.0) -> float:
    """Replace NaN / None with a safe fallback value."""
    if value is None:
        return fallback
    try:
        f = float(value)
        return fallback if math.isnan(f) else f
    except (TypeError, ValueError):
        return fallback


async def fetch_open_meteo(req: ForecastRequest) -> dict:
    """
    Fetch 48-hour historical baseline + target day forecast from both
    Open-Meteo Marine API and Weather API simultaneously.
    Returns a merged hourly dictionary keyed by ISO timestamp strings.
    """
    forecast_days = _resolve_forecast_days(req.target_date)

    common_params = {
        "latitude": req.latitude,
        "longitude": req.longitude,
        "past_days": 2,            # 48-hour historical baseline
        "forecast_days": forecast_days,
        "timezone": "auto",        # Critical: sync to user's actual timezone
    }

    marine_params = {**common_params, "hourly": MARINE_VARS}
    weather_params = {**common_params, "hourly": WEATHER_VARS}

    async with httpx.AsyncClient(timeout=30.0) as client:
        marine_resp, weather_resp = await asyncio.gather(
            client.get(MARINE_API, params=marine_params),
            client.get(WEATHER_API, params=weather_params),
        )

    if marine_resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo Marine API error: {marine_resp.text[:200]}"
        )
    if weather_resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo Weather API error: {weather_resp.text[:200]}"
        )

    marine_data = marine_resp.json()
    weather_data = weather_resp.json()

    # Merge both hourly datasets by timestamp
    marine_hourly = marine_data.get("hourly", {})
    weather_hourly = weather_data.get("hourly", {})
    timestamps = marine_hourly.get("time", [])

    merged = {}
    for i, ts in enumerate(timestamps):
        merged[ts] = {
            # --- Marine Variables ---
            "wave_height":          _safe(marine_hourly.get("wave_height", [None])[i] if i < len(marine_hourly.get("wave_height", [])) else None),
            "wave_period":          _safe(marine_hourly.get("wave_period", [None])[i] if i < len(marine_hourly.get("wave_period", [])) else None),
            "swell_wave_height":    _safe(marine_hourly.get("swell_wave_height", [None])[i] if i < len(marine_hourly.get("swell_wave_height", [])) else None),
            "swell_wave_period":    _safe(marine_hourly.get("swell_wave_period", [None])[i] if i < len(marine_hourly.get("swell_wave_period", [])) else None),
            "swell_wave_direction": _safe(marine_hourly.get("swell_wave_direction", [None])[i] if i < len(marine_hourly.get("swell_wave_direction", [])) else None),
            "wind_wave_height":     _safe(marine_hourly.get("wind_wave_height", [None])[i] if i < len(marine_hourly.get("wind_wave_height", [])) else None),
            "wind_wave_period":     _safe(marine_hourly.get("wind_wave_period", [None])[i] if i < len(marine_hourly.get("wind_wave_period", [])) else None),
            # --- Weather Variables ---
            "wind_speed_10m":       _safe(weather_hourly.get("wind_speed_10m", [None])[i] if i < len(weather_hourly.get("wind_speed_10m", [])) else None),
            "wind_direction_10m":   _safe(weather_hourly.get("wind_direction_10m", [None])[i] if i < len(weather_hourly.get("wind_direction_10m", [])) else None),
            "wind_gusts_10m":       _safe(weather_hourly.get("wind_gusts_10m", [None])[i] if i < len(weather_hourly.get("wind_gusts_10m", [])) else None),
            "pressure_msl":         _safe(weather_hourly.get("pressure_msl", [None])[i] if i < len(weather_hourly.get("pressure_msl", [])) else None, fallback=1013.25),
            "temperature_2m":       _safe(weather_hourly.get("temperature_2m", [None])[i] if i < len(weather_hourly.get("temperature_2m", [])) else None),
        }

    return merged


# ===========================================================================
# SECTION 2 — PHYSICS & MATHEMATICAL ENGINE
# ===========================================================================

def classify_wind_angle(wind_dir: float, beach_orientation: int) -> str:
    """Classify wind direction relative to beach as Onshore / Offshore / Sideshore."""
    angle_diff = abs(wind_dir - beach_orientation)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    if angle_diff < 45:
        return "Onshore"
    elif angle_diff > 135:
        return "Offshore"
    else:
        return "Sideshore"


def compute_hour_metrics(row: dict, beach_orientation: int) -> dict:
    """
    Apply full physics engine to a single hourly data row.
    All derived values are pre-computed here — Gemini never does math.
    """
    wh  = row["wave_height"]
    wp  = row["wave_period"]
    swh = row["swell_wave_height"]
    swp = row["swell_wave_period"]
    ws  = row["wind_speed_10m"]
    wg  = row["wind_gusts_10m"]
    wd  = row["wind_direction_10m"]

    # A. Wind Conversions
    wind_knots   = ws * 1.943844
    gust_kmh     = wg * 3.6

    # B. Total Wave Power (Tension / Lead Hold Indicator)
    wave_power   = 0.49 * (wh ** 2) * wp

    # C. Swell Dynamics
    swell_speed  = swp * 5.6         # km/h
    swell_power  = 0.49 * (swh ** 2) * swp

    # D. Angle Classification
    wind_type    = classify_wind_angle(wd, beach_orientation)

    return {
        **row,
        "wind_knots":   round(wind_knots, 2),
        "gust_kmh":     round(gust_kmh, 2),
        "wave_power":   round(wave_power, 3),
        "swell_speed":  round(swell_speed, 2),
        "swell_power":  round(swell_power, 3),
        "wind_type":    wind_type,
    }


def flag_hour(metrics: dict) -> str:
    """
    Evaluate a computed hour against safety thresholds.
    Returns 'RED', 'GREEN', or 'YELLOW'.
    """
    red_conditions = (
        metrics["wave_power"]  > 3.0  or
        metrics["wave_height"] > 1.8  or
        metrics["gust_kmh"]    > 50.0 or
        metrics["pressure_msl"] < 1005.0
    )
    green_conditions = (
        0.3  <= metrics["wave_height"] <= 1.0 and
        0.1  <= metrics["wave_power"]  <= 1.5 and
        metrics["wind_knots"] < 15.0
    )
    if red_conditions:
        return "RED"
    if green_conditions:
        return "GREEN"
    return "YELLOW"


def detect_temp_delta_flag(temp_series: list) -> bool:
    """
    Detect rapid cold front passage: temperature drop of >= 2.5°C
    within any 3-hour rolling window in the target day series.
    """
    for i in range(3, len(temp_series)):
        delta = temp_series[i] - temp_series[i - 3]
        if delta <= -2.5:
            return True
    return False


# ===========================================================================
# SECTION 3 — DATA AGGREGATION & TOKEN COMPRESSION
# ===========================================================================

DIURNAL_BLOCKS = {
    "Morning":      range(4, 12),    # 04:00–11:59
    "Afternoon":    range(12, 18),   # 12:00–17:59
    "Evening_Night": list(range(18, 24)) + list(range(0, 4)),  # 18:00–03:59
}


def aggregate_data(
    merged_hourly: dict,
    beach_orientation: int,
    beach_type: str,
    target_date: str
) -> dict:
    """
    Compress the full 72-hour time-series into a structured, token-minimal
    summary payload ready for Gemini API injection.
    """

    all_timestamps = sorted(merged_hourly.keys())

    # -----------------------------------------------------------------------
    # Resolve target date string → actual calendar date
    # -----------------------------------------------------------------------
    today = datetime.now(timezone.utc).date()
    date_offsets = {"today": 0, "tomorrow": 1, "day_after": 2}
    target_day = today + timedelta(days=date_offsets[target_date])
    target_day_str = target_day.isoformat()   # "YYYY-MM-DD"

    # -----------------------------------------------------------------------
    # Partition timestamps
    # -----------------------------------------------------------------------
    past_48h_ts  = [ts for ts in all_timestamps if ts[:10] < target_day_str]
    target_day_ts = [ts for ts in all_timestamps if ts[:10] == target_day_str]

    # -----------------------------------------------------------------------
    # Apply physics engine to every hour
    # -----------------------------------------------------------------------
    computed = {ts: compute_hour_metrics(merged_hourly[ts], beach_orientation)
                for ts in all_timestamps}

    # -----------------------------------------------------------------------
    # 1. PAST 48-HOUR BASELINE SUMMARY
    # -----------------------------------------------------------------------
    past_data    = [computed[ts] for ts in past_48h_ts]

    avg_wave_power_48h = (
        statistics.mean(h["wave_power"] for h in past_data) if past_data else 0.0
    )
    avg_swell_height_48h = (
        statistics.mean(h["swell_wave_height"] for h in past_data) if past_data else 0.0
    )
    avg_wave_period_48h = (
        statistics.mean(h["wave_period"] for h in past_data) if past_data else 0.0
    )

    # Dominant wind type over past 48 h (mode)
    wind_types_48h = [h["wind_type"] for h in past_data] if past_data else ["Unknown"]
    dominant_wind_type = max(set(wind_types_48h), key=wind_types_48h.count)

    # Sustained wind (knots > 15 = sustained threshold) duration in hours
    sustained_hours = sum(1 for h in past_data if h["wind_knots"] > 15)

    baseline_summary = {
        "avg_wave_power_48h":    round(avg_wave_power_48h, 3),
        "avg_swell_height_48h":  round(avg_swell_height_48h, 2),
        "avg_wave_period_48h":   round(avg_wave_period_48h, 2),
        "dominant_wind_type":    dominant_wind_type,
        "sustained_wind_hours":  sustained_hours,
    }

    # -----------------------------------------------------------------------
    # 2. DIURNAL BLOCK AVERAGES (Target Day)
    # -----------------------------------------------------------------------
    target_data = [computed[ts] for ts in target_day_ts]

    def block_avg(ts_list, hour_range):
        block = [computed[ts] for ts in ts_list
                 if int(ts[11:13]) in hour_range]
        if not block:
            return None
        keys = ["wave_height", "wave_power", "wind_knots", "gust_kmh",
                "swell_power", "pressure_msl", "temperature_2m"]
        return {k: round(statistics.mean(h[k] for h in block), 3) for k in keys} | {
            "dominant_wind_type": max(
                set(h["wind_type"] for h in block),
                key=[h["wind_type"] for h in block].count
            )
        }

    diurnal_blocks = {
        block: block_avg(target_day_ts, hours)
        for block, hours in DIURNAL_BLOCKS.items()
    }

    # -----------------------------------------------------------------------
    # 3. MICRO-HOURLY FLAGS (Target Day)
    # -----------------------------------------------------------------------
    flagged_hours = {"RED": [], "GREEN": [], "YELLOW": []}
    for ts in target_day_ts:
        hour_flag = flag_hour(computed[ts])
        flagged_hours[hour_flag].append({
            "time":        ts[11:16],     # "HH:MM"
            "wave_height": computed[ts]["wave_height"],
            "wave_power":  computed[ts]["wave_power"],
            "wind_knots":  computed[ts]["wind_knots"],
            "gust_kmh":    computed[ts]["gust_kmh"],
            "pressure_msl": computed[ts]["pressure_msl"],
            "wind_type":   computed[ts]["wind_type"],
        })

    # Temperature delta red flag check
    temp_series = [computed[ts]["temperature_2m"] for ts in target_day_ts]
    cold_front_detected = detect_temp_delta_flag(temp_series)
    if cold_front_detected:
        # Append meta warning to RED flags
        flagged_hours["RED"].append({
            "time": "FULL_DAY",
            "warning": "Cold front passage detected (ΔT ≤ -2.5°C / 3h window)"
        })

    # -----------------------------------------------------------------------
    # 4. WEED RISK ASSESSMENT
    # -----------------------------------------------------------------------
    weed_risk = (
        avg_wave_period_48h >= 8.0 and
        avg_swell_height_48h > 1.0 and
        dominant_wind_type == "Onshore"
    )

    # -----------------------------------------------------------------------
    # 5. LEAD SINKER HOLD SUMMARY (using target day max wave power)
    # -----------------------------------------------------------------------
    if target_data:
        max_wave_power = max(h["wave_power"] for h in target_data)
        avg_wave_power_today = statistics.mean(h["wave_power"] for h in target_data)
    else:
        max_wave_power = avg_wave_power_today = 0.0

    # Weight recommendation logic (empirical surfcasting thresholds):
    if max_wave_power < 0.5:
        sinker_recommendation = "60–80g (Light conditions, minimal drag)"
    elif max_wave_power < 1.5:
        sinker_recommendation = "80–120g (Moderate tension, standard hold)"
    elif max_wave_power < 3.0:
        sinker_recommendation = "120–175g (Strong tension, spiked anchor lead advised)"
    else:
        sinker_recommendation = "175g+ OR session abort — lead will not hold safely"

    # -----------------------------------------------------------------------
    # 6. PROBABILISTIC BIOLOGICAL PREDICTION
    # -----------------------------------------------------------------------
    avg_temp_today = (
        statistics.mean(h["temperature_2m"] for h in target_data)
        if target_data else 15.0
    )
    current_month  = target_day.month

    species_predictions = []

    if avg_temp_today < 16.0 and dominant_wind_type == "Onshore":
        species_predictions.append({
            "species": "European Sea Bass (Loup / Bar)",
            "probability": "HIGH",
            "trigger": "Cold water + Onshore churn creates baitfish compression",
            "bait": "Live sand eel, fresh squid strip, or peeler crab"
        })
        species_predictions.append({
            "species": "Large White Seabream (Grand Sarg)",
            "probability": "HIGH",
            "trigger": "Cold + turbid water activates bottom foraging",
            "bait": "Sea worm (ragworm/lugworm), mussel, or fresh limpet"
        })

    if avg_temp_today > 19.0 and dominant_wind_type in ("Offshore", "Sideshore"):
        species_predictions.append({
            "species": "Gilthead Seabream (Daurade Royale)",
            "probability": "HIGH",
            "trigger": "Warm, clear, calm sea — Daurade enters shallow structure",
            "bait": "Shore crab, small cuttlefish strip, or prawn"
        })
        species_predictions.append({
            "species": "Striped Seabream (Marbré)",
            "probability": "HIGH",
            "trigger": "Warm flat sea + sandy/rocky mixed seabed",
            "bait": "Ragworm, small prawn, or mussel"
        })

    if beach_type == "rocky" and any(
        h["wave_height"] > 0.4 for h in target_data
    ):
        if not any(s["species"] == "Large White Seabream (Grand Sarg)"
                   for s in species_predictions):
            species_predictions.append({
                "species": "White Seabream (Sarg)",
                "probability": "MEDIUM-HIGH",
                "trigger": "Rocky ground + foam line = prime Sarg ambush corridor",
                "bait": "Sea urchin, mussels, or ragworm on rocky bottom rig"
            })

    if beach_type == "sandy" and dominant_wind_type in ("Offshore", "Sideshore"):
        species_predictions.append({
            "species": "Flathead Mullet (Bouri) / Striped Seabream (Marbré)",
            "probability": "MEDIUM",
            "trigger": "Flat sandy surf creates shallow feeding lanes",
            "bait": "Bread dough (Mullet), ragworm or small prawn (Marbré)"
        })

    # Seasonal bonus (October–February: Sarg + Bass peak)
    if current_month in (10, 11, 12, 1, 2) and not any(
        s["species"] == "European Sea Bass (Loup / Bar)" for s in species_predictions
    ):
        species_predictions.append({
            "species": "European Sea Bass (Loup / Bar)",
            "probability": "MEDIUM (Seasonal Peak Window)",
            "trigger": "Winter months trigger feeding aggression pre-migration",
            "bait": "Fresh sardine fillet, squid, or large ragworm"
        })

    if not species_predictions:
        species_predictions.append({
            "species": "Opportunistic Mixed Bag (Bogue, Mullet, Sarg)",
            "probability": "LOW-MEDIUM",
            "trigger": "Transitional conditions — no dominant species trigger active",
            "bait": "Ragworm or small crab pieces"
        })

    # -----------------------------------------------------------------------
    # FINAL COMPRESSED PAYLOAD
    # -----------------------------------------------------------------------
    return {
        "meta": {
            "latitude":          req.latitude if hasattr(req, 'latitude') else None,
            "longitude":         req.longitude if hasattr(req, 'longitude') else None,
            "target_date":       target_day_str,
            "beach_orientation": beach_orientation,
            "beach_type":        beach_type,
        },
        "past_48h_baseline":    baseline_summary,
        "weed_risk_active":     weed_risk,
        "diurnal_blocks":       diurnal_blocks,
        "flagged_hours":        flagged_hours,
        "cold_front_detected":  cold_front_detected,
        "sinker_recommendation": sinker_recommendation,
        "max_wave_power_today": round(max_wave_power, 3),
        "avg_wave_power_today": round(avg_wave_power_today, 3),
        "avg_temp_today":       round(avg_temp_today, 1),
        "species_predictions":  species_predictions,
    }


# ===========================================================================
# SECTION 4 — GEMINI API SYNTHESIS
# ===========================================================================

SYSTEM_PROMPT = """You are an elite Surfcasting Master and Marine Analytics Expert. Your sole responsibility is to translate the provided pre-calculated physical variables, historical baselines, and biological probabilities into a highly practical tactical fishing report.

Do NOT perform math. All numbers are final. Your role is pure contextual synthesis and tactical interpretation.

You must format your output into these 4 clean sections, using the exact section headers below:

## 1. 🌊 Marine & Historical Context
Shrewdly link the past 48h sea state (wave power baseline, dominant wind type, sustained wind duration, weed risk flag) to today's expected water clarity, bottom turbulence, and weed presence. Explain what the angler will actually encounter at the water's edge.

## 2. 🕐 Diurnal Block Shifts
Outline the day's full progression across Morning (04:00–11:00), Afternoon (12:00–17:00), and Evening/Night (18:00–03:00). Alert the angler to critical transitions in wave power, wind type shifts, and pressure changes between blocks. Make it immediately actionable.

## 3. ⚡ Hourly Tactical Windows & Rig Strategy
Pinpoint the exact GREEN flag hours as prime fishing windows. Clearly flag RED hours as hazardous or unproductive. State explicitly whether the sinker weight recommendation will hold based on the wave tension data provided, and justify it. Provide rig setup advice that directly matches the sea state (hook size, leader strength, casting angle relative to wind type).

## 4. 🎯 Probabilistic Catch Forecasting
Present the species forecast ranked by probability. For each species: explain the ecological trigger driving its presence, specify the optimal bait presentation, and suggest the ideal time block for targeting it. Keep tone peer-to-peer — direct, data-backed, zero filler.

Tone: Professional, grounded in marine physics, peer-to-peer with an experienced angler. No generic disclaimers."""


def build_gemini_context(payload: dict) -> str:
    """
    Serialize the compressed Python analytics payload into a clean,
    token-efficient natural language data block for Gemini injection.
    """
    p = payload
    b = p["past_48h_baseline"]
    d = p["diurnal_blocks"]
    f = p["flagged_hours"]

    lines = [
        "=== SURFCASTING ANALYTICS DATA PACKET ===",
        f"Location: {p['meta']['latitude']}°N, {p['meta']['longitude']}°E",
        f"Target Date: {p['meta']['target_date']}",
        f"Beach Orientation: {p['meta']['beach_orientation']}° | Type: {p['meta']['beach_type'].upper()}",
        "",
        "--- PAST 48H BASELINE ---",
        f"Average Wave Power (48h): {b['avg_wave_power_48h']} kW/m",
        f"Average Swell Height (48h): {b['avg_swell_height_48h']} m",
        f"Average Wave Period (48h): {b['avg_wave_period_48h']} s",
        f"Dominant Wind Type (48h): {b['dominant_wind_type']}",
        f"Sustained Wind Hours (>15 knots): {b['sustained_wind_hours']}h",
        f"Weed Risk Active: {'YES — High weed load expected' if p['weed_risk_active'] else 'NO — Low weed risk'}",
        "",
        "--- TARGET DAY OVERVIEW ---",
        f"Average Air Temp: {p['avg_temp_today']}°C",
        f"Cold Front Detected: {'YES — Rapid temp drop (ΔT ≤ -2.5°C)' if p['cold_front_detected'] else 'NO'}",
        f"Max Wave Power Today: {p['max_wave_power_today']} kW/m",
        f"Average Wave Power Today: {p['avg_wave_power_today']} kW/m",
        f"Sinker Weight Recommendation: {p['sinker_recommendation']}",
        "",
        "--- DIURNAL BLOCK AVERAGES ---",
    ]

    for block_name, block_data in d.items():
        if block_data:
            lines.append(f"  [{block_name.replace('_', '/')}]")
            for k, v in block_data.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  [{block_name}]: No data available")

    lines.append("")
    lines.append("--- HOURLY FLAGS (TARGET DAY) ---")

    for flag_color in ("RED", "GREEN", "YELLOW"):
        hours = f[flag_color]
        if hours:
            lines.append(f"  {flag_color} FLAG HOURS ({len(hours)} total):")
            for h in hours[:10]:  # Cap at 10 entries per flag to control tokens
                hour_str = h.get("time", "N/A")
                if "warning" in h:
                    lines.append(f"    {hour_str}: {h['warning']}")
                else:
                    lines.append(
                        f"    {hour_str} → H={h.get('wave_height','?')}m | "
                        f"Power={h.get('wave_power','?')}kW/m | "
                        f"Wind={h.get('wind_knots','?')}kts | "
                        f"Gust={h.get('gust_kmh','?')}km/h | "
                        f"P={h.get('pressure_msl','?')}hPa | "
                        f"Type={h.get('wind_type','?')}"
                    )

    lines.append("")
    lines.append("--- BIOLOGICAL PROBABILITY MATRIX ---")
    for sp in p["species_predictions"]:
        lines.append(f"  Species: {sp['species']}")
        lines.append(f"    Probability: {sp['probability']}")
        lines.append(f"    Ecological Trigger: {sp['trigger']}")
        lines.append(f"    Bait Recommendation: {sp['bait']}")

    lines.append("")
    lines.append("=== END OF DATA PACKET ===")
    lines.append("Now generate the full tactical report using ONLY the data above.")

    return "\n".join(lines)


async def call_gemini(context_block: str) -> str:
    """
    Send the pre-compressed analytics context to Gemini for natural language synthesis.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set."
        )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.65,       # Balanced: creative but grounded
            max_output_tokens=2048,
            top_p=0.9,
        )
    )

    response = await model.generate_content_async(context_block)
    return response.text


# ===========================================================================
# SECTION 5 — API ENDPOINT
# ===========================================================================

import asyncio  # noqa: E402 (placed after definitions for readability)


@app.get("/health")
async def health_check():
    """Lightweight health probe for Render uptime monitoring."""
    return {"status": "operational", "service": "SurfcastAPI v1.0"}


@app.post("/analyze")
async def analyze(req: ForecastRequest):
    """
    Main endpoint. Orchestrates:
    1. Open-Meteo data fetch (parallel Marine + Weather)
    2. Full physics engine pass on every hour
    3. Token-compressed aggregation
    4. Gemini AI synthesis
    5. Returns structured JSON with raw analytics + AI narrative
    """
    # Step 1: Fetch raw data
    merged_hourly = await fetch_open_meteo(req)

    if not merged_hourly:
        raise HTTPException(status_code=502, detail="No hourly data returned from Open-Meteo.")

    # Step 2–4: Calculate physics + aggregate + compress
    # Note: Pass req metadata into aggregate_data for biological predictions
    payload = aggregate_data(
        merged_hourly,
        req.beach_orientation,
        req.beach_type,
        req.target_date
    )
    # Inject req metadata properly (aggregate_data uses local scope)
    payload["meta"]["latitude"]  = req.latitude
    payload["meta"]["longitude"] = req.longitude

    # Step 5: Build Gemini context block
    context_block = build_gemini_context(payload)

    # Step 6: Call Gemini for synthesis
    ai_narrative = await call_gemini(context_block)

    # Step 7: Return full structured response
    return {
        "analytics": payload,           # Raw pre-calculated data (for frontend cards)
        "ai_report":  ai_narrative,     # Gemini's tactical narrative
        "context_tokens_approx": len(context_block.split()),  # Debug token audit
                                                            }
