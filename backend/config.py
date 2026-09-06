"""
config.py – إعدادات مركزية لتطبيق Surfcasting Analytics (v22).

الهدف: كل ما كان ثابتاً (Hard-coded) داخل main.py صار قابلاً للضبط من
متغيّرات البيئة (Environment Variables) بدون تعديل الكود:

  GEMINI_API_KEY              مفتاح Gemini (اختياري عند الإقلاع)
  GEMINI_MODEL                اسم النموذج (افتراضي gemini-2.5-flash)
  GEMINI_THINKING_BUDGET      ميزانية التفكير (0 = تعطيل التفكير = أسرع)
  GEMINI_MAX_OUTPUT_TOKENS    أقصى طول للتقرير
  GEMINI_READ_TIMEOUT_S       مهلة قراءة الرد من Gemini
  REPORT_TIMEOUT_S            المهلة الكلية لتوليد التقرير
  CACHE_ENABLED               تفعيل الكاش الذاكري
  CACHE_TTL_REPORT_S          زمن صلاحية التقرير المولّد
  CACHE_TTL_UPSTREAM_S        زمن صلاحية بيانات Open-Meteo
  CACHE_TTL_OVERPASS_S        زمن صلاحية نتائج Overpass (طويل: الخرائط ثابتة)
  CACHE_TTL_FAILURE_S         زمن تذكّر الفشل (يمنع ضرب الخدمات المريضة)
  CACHE_MAX_ENTRIES           أقصى عدد مفاتيح لكل كاش
  RATE_LIMIT_ENABLED          تفعيل/تعطيل تحديد المعدّل (مفيد للاختبارات)
  RATE_LIMIT_REPORT           معدّل السماح لتوليد التقرير
  ALLOWED_ORIGINS             قائمة Origins المسموحة (CORS)
  LOG_LEVEL                   مستوى السجلات
"""

from __future__ import annotations

import os
from typing import List

TRUE_VALUES = {"1", "true", "yes", "on", "y", "oui"}
FALSE_VALUES = {"0", "false", "no", "off", "n", ""}


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def env_float_list(name: str, default: List[float]) -> List[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    out: List[float] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            continue
    return out or list(default)


class Settings:
    """قيم الإعدادات محسوبة مرة واحدة عند استيراد الوحدة."""

    APP_NAME = "Surfcasting Analytics"
    VERSION = "22.0.0"
    DEFAULT_TZ = env_str("DEFAULT_TZ", "Africa/Tunis")

    # ---------- Gemini ----------
    GEMINI_API_KEY = env_str("GEMINI_API_KEY", "")
    GEMINI_MODEL = env_str("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_URL = env_str(
        "GEMINI_URL",
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
    )
    GEMINI_TEMPERATURE = env_float("GEMINI_TEMPERATURE", 0.1)
    GEMINI_MAX_OUTPUT_TOKENS = env_int("GEMINI_MAX_OUTPUT_TOKENS", 8000)
    # 0 = إيقاف التفكير العميق (أسرع بكثير مع 2.5-flash). -1 = ترك إعداد النموذج الافتراضي.
    GEMINI_THINKING_BUDGET = env_int("GEMINI_THINKING_BUDGET", 0)
    GEMINI_CONNECT_TIMEOUT_S = env_float("GEMINI_CONNECT_TIMEOUT_S", 10.0)
    GEMINI_READ_TIMEOUT_S = env_float("GEMINI_READ_TIMEOUT_S", 90.0)
    GEMINI_RETRY_WAITS = env_float_list("GEMINI_RETRY_WAITS", [3.0, 7.0])
    GEMINI_MAX_ATTEMPTS = env_int("GEMINI_MAX_ATTEMPTS", 3)
    GEMINI_MAX_CONTEXT_CHARS = env_int("GEMINI_MAX_CONTEXT_CHARS", 30000)
    # عند تعذّر Gemini (غياب المفتاح، ضغط API، نفاد الحصة) نبني التقرير محلياً
    OFFLINE_FALLBACK = env_bool("OFFLINE_FALLBACK", True)

    # ---------- الوقت ----------
    REPORT_TIMEOUT_S = env_float("REPORT_TIMEOUT_S", 140.0)
    UPSTREAM_TIMEOUT_S = env_float("UPSTREAM_TIMEOUT_S", 20.0)
    OVERPASS_TIMEOUT_S = env_float("OVERPASS_TIMEOUT_S", 25.0)

    # ---------- الكاش ----------
    CACHE_ENABLED = env_bool("CACHE_ENABLED", True)
    CACHE_TTL_REPORT_S = env_float("CACHE_TTL_REPORT_S", 600.0)
    CACHE_TTL_UPSTREAM_S = env_float("CACHE_TTL_UPSTREAM_S", 900.0)
    CACHE_TTL_OVERPASS_S = env_float("CACHE_TTL_OVERPASS_S", 86400.0)
    CACHE_TTL_FAILURE_S = env_float("CACHE_TTL_FAILURE_S", 60.0)
    CACHE_MAX_ENTRIES = env_int("CACHE_MAX_ENTRIES", 512)
    CACHE_SERVE_STALE_ON_ERROR = env_bool("CACHE_SERVE_STALE_ON_ERROR", True)

    # ---------- تحديد المعدّل ----------
    RATE_LIMIT_ENABLED = env_bool("RATE_LIMIT_ENABLED", True)
    RATE_LIMIT_REPORT = env_str("RATE_LIMIT_REPORT", "6/minute")
    RATE_LIMIT_ORIENTATION = env_str("RATE_LIMIT_ORIENTATION", "5/minute")
    RATE_LIMIT_BOTTOM = env_str("RATE_LIMIT_BOTTOM", "10/minute")

    # ---------- متفرقات ----------
    ALLOWED_ORIGINS = env_str("ALLOWED_ORIGINS", "*")
    LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()

    @property
    def gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY)


settings = Settings()
