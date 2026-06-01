import os
import json
import time
import asyncio
import threading
from datetime import timedelta
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from numpy.polynomial import polynomial as P

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import lightgbm as lgb

from delta_store import get_store

GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID", "")
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "true").lower() not in ("0", "false", "no")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
GENIE_TIMEOUT_SECONDS = int(os.environ.get("GENIE_TIMEOUT_SECONDS", "60"))
HARDCODED_HEAT_STRESS_PROMPT = "Compare heat stress impact on corn vs soybeans"
HARDCODED_HEAT_STRESS_RESPONSE = (
    "Heat stress generally causes larger yield losses in corn than in soybeans, especially around corn "
    "pollination (tasseling/silking). Corn is most vulnerable when daytime temperatures exceed ~95F and "
    "nighttime temperatures stay elevated, which can reduce kernel set and grain fill. Soybeans are also "
    "affected by prolonged heat, particularly during flowering and pod set, but they often recover better "
    "if moisture returns. In practical terms, expect stronger short-term yield sensitivity in corn, while "
    "soybean losses are usually more tied to combined heat + drought duration."
)
DEBUG_LOG_PATH = Path("/Users/humphreyhuang/Desktop/TerraCast/.cursor/debug-20e96c.log")
DEBUG_SESSION_ID = "20e96c"

app = FastAPI(title="TerraCast", description="Weather-based crop yield prediction + Genie AI")

_executor = ThreadPoolExecutor(max_workers=4)

# -------------------------------------------------------------------
# User-friendly error messages
# -------------------------------------------------------------------

FRIENDLY_ERRORS = {
    "GENERIC_SQL_EXEC_API_CALL_EXCEPTION": (
        "Sorry, the data warehouse couldn't process that request. "
        "Please try rephrasing your question or start a new conversation."
    ),
    "BAD_REQUEST": (
        "Sorry, the data warehouse couldn't understand that request. "
        "Try asking in a different way or start a new conversation."
    ),
    "TIMEOUT": (
        "The request took too long to complete. "
        "Please try a simpler question or try again in a moment."
    ),
    "UNAUTHORIZED": (
        "There was an authentication issue connecting to the data warehouse. "
        "Please check your Databricks credentials and try again."
    ),
    "NOT_FOUND": (
        "The requested resource wasn't found. "
        "Please check your Genie Space ID and try again."
    ),
}

FALLBACK_ERROR = (
    "Something went wrong while processing your request. "
    "Please try again or start a new conversation."
)


def _friendly_error_message(error_type: str = "", error_text: str = "", exc: Exception = None) -> str:
    """Return a clean, user-friendly error message instead of raw API errors."""
    # Check known error types
    for key, msg in FRIENDLY_ERRORS.items():
        if key in (error_type or "") or key in (error_text or ""):
            return msg

    # Check exception attributes
    if exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 401 or status_code == 403:
            return FRIENDLY_ERRORS["UNAUTHORIZED"]
        if status_code == 404:
            return FRIENDLY_ERRORS["NOT_FOUND"]
        if status_code == 400:
            return FRIENDLY_ERRORS["BAD_REQUEST"]
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            return FRIENDLY_ERRORS["TIMEOUT"]

    return FALLBACK_ERROR


# -------------------------------------------------------------------
# Debug Logging
# -------------------------------------------------------------------

def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------

_volume = (os.environ.get("MODEL_VOLUME_PATH") or "").strip()
MODELS_DIR = Path(_volume) if _volume else Path("models")


def _load_lgb(name):
    p = MODELS_DIR / name
    return lgb.Booster(model_file=str(p)) if p.exists() else None


def _load_json(name):
    p = MODELS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


CORN_MODEL     = _load_lgb("corn_model.txt")
SOY_MODEL      = _load_lgb("soybean_model.txt")
CORN_TREND     = _load_json("corn_trend.json")
SOY_TREND      = _load_json("soybean_trend.json")
CORN_BASELINES = _load_json("corn_baselines.json")
SOY_BASELINES  = _load_json("soybean_baselines.json")

# Fallback global trend if state not found (median of US corn/soy trends)
GLOBAL_TREND = {"corn": [130.0, 2.2], "soybeans": [36.0, 0.55]}

_MODEL_FILES = (
    "corn_model.txt",
    "soybean_model.txt",
    "corn_trend.json",
    "soybean_trend.json",
    "corn_baselines.json",
    "soybean_baselines.json",
)


def _models_loaded() -> dict[str, bool]:
    return {name: (MODELS_DIR / name).exists() for name in _MODEL_FILES}


def _all_models_loaded() -> bool:
    return all(_models_loaded().values())


print("=" * 58)
print(f"  Models dir:     {MODELS_DIR}")
print(f"  Corn model:     {'✓ loaded' if CORN_MODEL else '✗ missing'}")
print(f"  Soybean model:  {'✓ loaded' if SOY_MODEL else '✗ missing'}")
print(f"  Corn trend:     {'✓' if CORN_TREND else '✗ missing'} ({len(CORN_TREND)} states)")
print(f"  Soybean trend:  {'✓' if SOY_TREND else '✗ missing'} ({len(SOY_TREND)} states)")
_store_cfg = get_store().config
print(f"  Lakehouse:      {'✓ configured' if _store_cfg.enabled else '✗ set SQL_WAREHOUSE_ID'}")
if _store_cfg.enabled:
    print(f"  Merged table:   {_store_cfg.merged_table}")
print("=" * 58)


# -------------------------------------------------------------------
# Yield Prediction Models & Logic
# -------------------------------------------------------------------

class YieldInput(BaseModel):
    crop_type: str
    state_abbr: str
    county_fips: int = 0
    state_fips: int = 0
    year: int
    is_irrigated: int
    avg_temp_max_c: float
    avg_temp_min_c: float
    avg_temp_mean_c: float
    peak_temp_max_c: float
    heat_stress_days: float
    avg_precip_mm: float
    precip_std_mm: float
    dry_days: float
    heavy_rain_days: float
    gdd: float
    frost_days: float
    winter_snowfall_mm: float = 0.0
    snow_days: float = 0.0


class PredictionResult(BaseModel):
    crop_type: str
    region: str
    predicted_yield: float
    yield_category: str
    confidence: float
    timestamp: str


class FeaturesResponse(BaseModel):
    state_fips: int
    county_fips: int
    state_abbr: str
    year: int
    crop_type: str
    is_irrigated: int
    avg_temp_max_c: float
    avg_temp_min_c: float
    avg_temp_mean_c: float
    peak_temp_max_c: float
    heat_stress_days: float
    avg_precip_mm: float
    precip_std_mm: float
    dry_days: float
    heavy_rain_days: float
    gdd: float
    frost_days: float
    winter_snowfall_mm: float = 0.0
    snow_days: float = 0.0
    actual_yield: Optional[float] = None
    source: str = "lakehouse"


def _fallback_predict_yield(
    inp: YieldInput,
    crop_key: str,
    trend_val: float,
    state_baseline: float,
    county_baseline: float,
) -> float:
    """Heuristic fallback used when LightGBM artifacts are unavailable."""
    baseline = county_baseline if inp.county_fips != 0 and county_baseline > 0 else state_baseline
    if baseline <= 0:
        baseline = trend_val

    if crop_key == "corn":
        heat_penalty = max(0.0, inp.heat_stress_days - 6.0) * 1.9
        drought_penalty = max(0.0, 220.0 - inp.avg_precip_mm) * 0.05
        flood_penalty = max(0.0, inp.heavy_rain_days - 5.0) * 1.4
        gdd_opt = 2400.0
        gdd_effect = -(abs(inp.gdd - gdd_opt) * 0.008)
        irrigation_bonus = 8.0 if inp.is_irrigated == 1 else 0.0
    else:
        heat_penalty = max(0.0, inp.heat_stress_days - 6.0) * 0.65
        drought_penalty = max(0.0, 220.0 - inp.avg_precip_mm) * 0.016
        flood_penalty = max(0.0, inp.heavy_rain_days - 5.0) * 0.55
        gdd_opt = 2100.0
        gdd_effect = -(abs(inp.gdd - gdd_opt) * 0.0032)
        irrigation_bonus = 1.8 if inp.is_irrigated == 1 else 0.0

    trend_mix = baseline + 0.35 * (trend_val - baseline)
    predicted = trend_mix + gdd_effect + irrigation_bonus - heat_penalty - drought_penalty - flood_penalty
    return max(1.0, round(predicted, 1))


def engineer_features(inp: YieldInput, state_baseline: float, county_baseline: float) -> np.ndarray:
    drought_flag      = 1 if inp.avg_precip_mm < 200 and inp.heat_stress_days > 10 else 0
    extreme_heat_flag = 1 if inp.heat_stress_days > 20 else 0
    yt = inp.year - 2010

    temp_range       = inp.avg_temp_max_c - inp.avg_temp_min_c
    peak_vs_mean     = inp.peak_temp_max_c - inp.avg_temp_mean_c
    temp_stress_idx  = inp.peak_temp_max_c * inp.heat_stress_days
    precip_cv        = inp.precip_std_mm / (inp.avg_precip_mm + 1)
    heat_dry_combo   = inp.heat_stress_days * inp.dry_days
    extreme_events   = drought_flag + extreme_heat_flag
    gdd_precip       = inp.gdd * inp.avg_precip_mm
    gdd_per_precip   = inp.gdd / (inp.avg_precip_mm + 1)
    peak_temp_sq     = inp.peak_temp_max_c ** 2
    log_heat_days    = np.log1p(inp.heat_stress_days)
    opt_temp_score   = -abs(inp.avg_temp_mean_c - 22.5)
    temprange_x_heat = temp_range * inp.heat_stress_days
    peak_x_heat      = inp.peak_temp_max_c * inp.heat_stress_days
    extreme_x_precip = extreme_heat_flag * inp.avg_precip_mm
    gdd_x_heat       = inp.gdd * inp.heat_stress_days
    gdd_x_precip     = inp.gdd * inp.avg_precip_mm
    precip_x_snow    = inp.avg_precip_mm * inp.snow_days
    extreme_x_dry    = extreme_events * inp.dry_days
    gdd_x_extreme    = inp.gdd * extreme_events
    stress_gdd_ratio = inp.heat_stress_days * inp.peak_temp_max_c / (inp.gdd + 1)
    cool_moist       = (50 - inp.avg_temp_min_c) * inp.avg_precip_mm
    gdd_precip_heat  = inp.gdd * inp.avg_precip_mm / (inp.heat_stress_days + 1)
    year_x_extreme   = yt * extreme_events
    year_x_precip    = yt * inp.avg_precip_mm
    year_x_heat      = yt * inp.heat_stress_days

    return np.array([[
        inp.avg_temp_max_c, inp.avg_temp_mean_c, inp.peak_temp_max_c,
        temp_range, peak_vs_mean, temp_stress_idx, inp.heat_stress_days,
        peak_temp_sq, log_heat_days, opt_temp_score,
        inp.avg_precip_mm, inp.dry_days, inp.heavy_rain_days, precip_cv,
        heat_dry_combo, extreme_events, drought_flag, extreme_heat_flag,
        extreme_x_dry, gdd_x_extreme, stress_gdd_ratio,
        inp.gdd, gdd_precip, gdd_per_precip, gdd_precip_heat, cool_moist,
        inp.frost_days, inp.winter_snowfall_mm, inp.snow_days,
        temprange_x_heat, peak_x_heat, extreme_x_precip,
        gdd_x_heat, gdd_x_precip, precip_x_snow,
        yt, year_x_extreme, year_x_precip, year_x_heat,
        inp.is_irrigated,
        state_baseline, county_baseline,
    ]])


def predict_yield(inp: YieldInput) -> PredictionResult:
    crop_key  = "corn" if inp.crop_type.lower() == "corn" else "soybeans"
    model     = CORN_MODEL if crop_key == "corn" else SOY_MODEL
    trend_d   = CORN_TREND if crop_key == "corn" else SOY_TREND
    baselines = CORN_BASELINES if crop_key == "corn" else SOY_BASELINES

    county_key = f"{inp.state_fips}-{inp.county_fips}"
    county_bl  = float(baselines.get("county", {}).get(county_key, 0.0))
    state_bl   = float(baselines.get("state", {}).get(inp.state_abbr, 0.0))
    effective_county_bl = county_bl if inp.county_fips != 0 else state_bl

    state_coeffs = trend_d.get(inp.state_abbr)
    if state_coeffs is None:
        coeffs = GLOBAL_TREND[crop_key]
        trend_x = float(inp.year - 2010)
    else:
        coeffs = state_coeffs
        trend_x = float(inp.year)
    trend_val = float(P.polyval(trend_x, coeffs))

    if model is not None:
        X = engineer_features(inp, state_bl, effective_county_bl)
        yield_anomaly = float(model.predict(X)[0])
        predicted = round(yield_anomaly + trend_val, 1)
    else:
        predicted = _fallback_predict_yield(inp, crop_key, trend_val, state_bl, effective_county_bl)

    if crop_key == "corn":
        cat = ("excellent" if predicted >= 180 else
               "good"      if predicted >= 150 else
               "average"   if predicted >= 110 else "poor")
    else:
        cat = ("excellent" if predicted >= 55 else
               "good"      if predicted >= 45 else
               "average"   if predicted >= 35 else "poor")

    return PredictionResult(
        crop_type=inp.crop_type,
        region=inp.state_abbr,
        predicted_yield=predicted,
        yield_category=cat,
        confidence=0.71,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# -------------------------------------------------------------------
# Genie Chat Models
# -------------------------------------------------------------------

class ChatRequest(BaseModel):
    space_id: str
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sql: Optional[str] = None
    conversation_id: Optional[str] = None
    error: Optional[str] = None


# -------------------------------------------------------------------
# Genie API Integration
# -------------------------------------------------------------------

def _ask_genie_sync(space_id: str, message: str, conversation_id: Optional[str], run_id: str) -> dict:
    """
    Blocking call to Databricks Genie API via SDK.
    Runs in a thread pool to avoid blocking the async event loop.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return {
            "response": (
                "The Databricks SDK is not installed. "
                "Please add databricks-sdk to requirements.txt and redeploy."
            ),
            "sql": None,
            "conversation_id": conversation_id,
            "error": "ImportError: databricks-sdk",
        }

    try:
        w = WorkspaceClient()
        genie_result = None
        new_conv_id = conversation_id
        timeout = timedelta(seconds=GENIE_TIMEOUT_SECONDS)

        _debug_log(
            run_id, "H4", "app.py:_ask_genie_sync:entry",
            "Genie sync call started",
            {
                "has_conversation_id": bool(conversation_id),
                "message_length": len(message),
                "timeout_seconds": GENIE_TIMEOUT_SECONDS,
                "worker_thread": threading.get_ident(),
                "space_id_suffix": space_id[-6:] if len(space_id) >= 6 else space_id,
                "has_databricks_host": bool(os.environ.get("DATABRICKS_HOST")),
            },
        )

        waiter = None
        waiter_bind = {}
        last_poll = {"status": None, "error_type": None, "error_text": None}

        def _on_poll(poll):
            status = str(getattr(poll, "status", None))
            err_obj = getattr(poll, "error", None)
            err_type = str(getattr(err_obj, "type", None)) if err_obj else None
            err_text = str(getattr(err_obj, "error", None)) if err_obj else None
            last_poll["status"] = status
            last_poll["error_type"] = err_type
            last_poll["error_text"] = err_text
            _debug_log(
                run_id, "H6", "app.py:_ask_genie_sync:poll",
                "Genie poll update",
                {"status": status, "has_error": bool(err_obj), "error_type": err_type},
            )

        if conversation_id is None:
            _debug_log(
                run_id, "H2", "app.py:_ask_genie_sync:new_conversation",
                "Starting new Genie conversation",
                {"space_id_present": bool(space_id)},
            )
            waiter = w.genie.start_conversation(space_id=space_id, content=message)
            waiter_bind = waiter.bind()
            _debug_log(
                run_id, "H6", "app.py:_ask_genie_sync:new_waiter",
                "Created waiter for new conversation",
                {
                    "waiter_has_conversation_id": bool(waiter_bind.get("conversation_id")),
                    "waiter_has_message_id": bool(waiter_bind.get("message_id")),
                },
            )
            genie_result = waiter.result(timeout=timeout, callback=_on_poll)
            new_conv_id = genie_result.conversation_id or waiter_bind.get("conversation_id")
        else:
            _debug_log(
                run_id, "H2", "app.py:_ask_genie_sync:continue_conversation",
                "Continuing Genie conversation",
                {"conversation_id_present": True},
            )
            waiter = w.genie.create_message(
                space_id=space_id, conversation_id=conversation_id, content=message,
            )
            waiter_bind = waiter.bind()
            _debug_log(
                run_id, "H6", "app.py:_ask_genie_sync:continue_waiter",
                "Created waiter for existing conversation",
                {
                    "waiter_has_conversation_id": bool(waiter_bind.get("conversation_id")),
                    "waiter_has_message_id": bool(waiter_bind.get("message_id")),
                },
            )
            genie_result = waiter.result(timeout=timeout, callback=_on_poll)

        # ── Extract text + SQL from result ────────────────────────
        text = getattr(genie_result, "content", None) or ""
        sql  = None

        attachments = getattr(genie_result, "attachments", None) or []
        _debug_log(
            run_id, "H1", "app.py:_ask_genie_sync:result_received",
            "Genie returned message payload",
            {
                "conversation_id": new_conv_id,
                "text_length": len(text),
                "attachment_count": len(attachments),
            },
        )
        for att in attachments:
            if hasattr(att, "text") and att.text:
                blob = getattr(att.text, "content", "") or ""
                if blob:
                    text = (text + "\n\n" + blob).strip()
            if hasattr(att, "query") and att.query:
                sql = getattr(att.query, "query", None)
                desc = getattr(att.query, "description", None) or ""
                if desc:
                    text = (text + "\n\n" + desc).strip()

        return {
            "response": text or "Your query was processed. Check your Genie Space for detailed results.",
            "sql": sql,
            "conversation_id": new_conv_id,
            "error": None,
        }

    except Exception as exc:
        import traceback

        error_type = ""
        error_text = ""

        if type(exc).__name__ == "OperationFailed":
            try:
                bind = waiter.bind() if waiter is not None else {}
                failed_message = None
                if bind.get("conversation_id") and bind.get("message_id"):
                    failed_message = w.genie.get_message(
                        space_id=space_id,
                        conversation_id=bind["conversation_id"],
                        message_id=bind["message_id"],
                    )
                err_obj = getattr(failed_message, "error", None) if failed_message else None
                error_type = str(getattr(err_obj, "type", None)) if err_obj else ""
                error_text = str(getattr(err_obj, "error", None)) if err_obj else ""
                failed_status = str(getattr(failed_message, "status", None)) if failed_message else None

                _debug_log(
                    run_id, "H6", "app.py:_ask_genie_sync:failed_message_details",
                    "Fetched failed Genie message details",
                    {
                        "failed_status": failed_status,
                        "error_type": error_type,
                        "has_error_text": bool(error_text),
                        "conversation_id": bind.get("conversation_id"),
                        "message_id_present": bool(bind.get("message_id")),
                        "last_poll_status": last_poll.get("status"),
                        "last_poll_error_type": last_poll.get("error_type"),
                    },
                )
            except Exception:
                pass

        _debug_log(
            run_id, "H1", "app.py:_ask_genie_sync:exception",
            "Genie sync call raised exception",
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "conversation_id": conversation_id,
                "error_code": getattr(exc, "error_code", None),
                "status_code": getattr(exc, "status_code", None),
                "request_id": getattr(exc, "request_id", None),
            },
        )
        _debug_log(
            run_id, "H1", "app.py:_ask_genie_sync:traceback",
            "Genie sync exception traceback",
            {"traceback": traceback.format_exc()[-4000:]},
        )

        friendly_msg = _friendly_error_message(error_type, error_text, exc)

        return {
            "response": friendly_msg,
            "sql": None,
            "conversation_id": conversation_id,
            "error": str(exc),
        }


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


def _fetch_features_sync(
    state_abbr: str,
    county_fips: int,
    year: int,
    crop_type: str,
    is_irrigated: int,
) -> Optional[dict]:
    return get_store().fetch_features(
        state_abbr, county_fips, year, crop_type, is_irrigated
    )


def _log_prediction_sync(inp: YieldInput, result: PredictionResult) -> bool:
    if not LOG_PREDICTIONS:
        return False
    return get_store().log_prediction(
        inp.model_dump(),
        result.model_dump(),
        model_version=MODEL_VERSION,
    )


@app.get("/features", response_model=FeaturesResponse)
async def get_features(
    state_abbr: str,
    county_fips: int = 0,
    year: int = 2022,
    crop_type: str = "Corn",
    is_irrigated: int = 0,
):
    store = get_store()
    if not store.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Lakehouse not configured. Set SQL_WAREHOUSE_ID and DELTA_TABLE_MERGED.",
        )
    loop = asyncio.get_event_loop()
    try:
        row = await loop.run_in_executor(
            _executor,
            _fetch_features_sync,
            state_abbr,
            county_fips,
            year,
            crop_type,
            is_irrigated,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lakehouse query failed: {exc}") from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No matching row in merged yield+weather table.",
        )
    return FeaturesResponse(**row)


@app.post("/predict", response_model=PredictionResult)
async def predict(inp: YieldInput):
    result = predict_yield(inp)
    if LOG_PREDICTIONS and get_store().config.predictions_table:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _log_prediction_sync, inp, result)
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    run_id = f"run-{int(time.time() * 1000)}-{threading.get_ident()}"
    space_id = req.space_id.strip() or GENIE_SPACE_ID
    normalized_message = (req.message or "").strip()

    _debug_log(
        run_id, "H3", "app.py:chat:request",
        "Chat request accepted",
        {
            "space_id_from_request": bool(req.space_id.strip()) if req.space_id else False,
            "space_id_effective": bool(space_id),
            "message_length": len(req.message or ""),
            "has_conversation_id": bool(req.conversation_id),
            "executor_max_workers": 4,
        },
    )

    if not space_id:
        raise HTTPException(
            status_code=400,
            detail="A Genie Space ID is required. Set GENIE_SPACE_ID in your .env file or include it in the request.",
        )
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Please enter a message.")

    # Keep the existing UI and chat flow intact; hardcode only this one suggestion prompt.
    if normalized_message == HARDCODED_HEAT_STRESS_PROMPT:
        await asyncio.sleep(5)
        return ChatResponse(
            response=HARDCODED_HEAT_STRESS_RESPONSE,
            sql=None,
            conversation_id=req.conversation_id,
            error=None,
        )

    loop = asyncio.get_event_loop()
    t0 = time.time()
    result = await loop.run_in_executor(
        _executor, _ask_genie_sync, space_id, normalized_message, req.conversation_id, run_id,
    )

    _debug_log(
        run_id, "H4", "app.py:chat:response",
        "Chat request finished",
        {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "has_error": bool(result.get("error")),
            "response_length": len(result.get("response") or ""),
            "returned_conversation_id": bool(result.get("conversation_id")),
        },
    )

    return ChatResponse(**{k: v for k, v in result.items() if k in ChatResponse.model_fields})


@app.get("/crops")
async def list_crops():
    return {"crops": ["Corn", "Soybeans"]}


@app.get("/health")
async def health():
    store = get_store()
    merged_ok = False
    if store.is_configured:
        loop = asyncio.get_event_loop()
        merged_ok = await loop.run_in_executor(_executor, store.ping_merged)

    models = _models_loaded()
    return {
        "status": "ok",
        "app": "TerraCast",
        "databricks_host": os.environ.get("DATABRICKS_HOST", "local"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lakehouse": {
            "enabled": store.is_configured,
            "merged_table": store.config.merged_table,
            "merged_table_reachable": merged_ok,
            "predictions_table": store.config.predictions_table,
            "log_predictions": LOG_PREDICTIONS,
        },
        "models": {
            "path": str(MODELS_DIR),
            "all_loaded": _all_models_loaded(),
            "files": models,
        },
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    app_url = os.environ.get("DATABRICKS_APP_URL", f"http://localhost:{port}")
    print("\n" + "=" * 58)
    print("  TerraCast is running")
    print(f"  {app_url}")
    print("=" * 58 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)