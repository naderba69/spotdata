"""
cache.py – كاش ذاكري بسيط بصلاحية زمنية (TTL) مع «رحلة واحدة» (Single-flight).

المشاكل التي يحلّها:
  1. الطلبات المتكررة لنفس البقعة تستهلك رصيد Gemini وتطيل زمن الانتظار
     → نخزّن التقرير النهائي لمدّة CACHE_TTL_REPORT_S.
  2. بيانات Open-Meteo تتغيّر كل ساعة فقط → نخزّنها CACHE_TTL_UPSTREAM_S.
  3. نتائج Overpass (اتجاه الشاطئ/نوع القاع) شبه ثابتة → نخزّنها يوماً كاملاً.
  4. ضغطات الزر المزدوجة أو الطلبات المتزامنة المتطابقة تولّد عدة نداءات
     Gemini في نفس اللحظة → آلية Single-flight تجعلها تنتظر نداءً واحداً.
  5. عند فشل خدمة خارجية نتذكّر الفشل لمدّة قصيرة بدل ضربها في كل طلب
     (Negative caching) مع إمكانية تقديم قيمة قديمة (Stale) إن وُجدت.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


@dataclass
class _Entry:
    value: Any
    created_at: float
    expires_at: float


class TTLCache:
    """كاش TTL آمن مع asyncio، يدعم get_or_set و single-flight."""

    def __init__(self, name: str, ttl: float, max_entries: int = 512,
                 enabled: bool = True, serve_stale_on_error: bool = True):
        self.name = name
        self.ttl = float(ttl)
        self.max_entries = int(max_entries)
        self.enabled = enabled
        self.serve_stale_on_error = serve_stale_on_error
        self._data: Dict[Any, _Entry] = {}
        self._inflight: Dict[Any, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._joined = 0

    # ---------- عمليات داخلية (تستدعى مع القفل) ----------
    def _purge_locked(self, now: float) -> None:
        expired = [k for k, e in self._data.items() if e.expires_at <= now]
        for k in expired:
            self._data.pop(k, None)
        if len(self._data) > self.max_entries:
            overflow = sorted(self._data.items(), key=lambda kv: kv[1].created_at)
            for k, _ in overflow[: len(self._data) - self.max_entries]:
                self._data.pop(k, None)

    def _store_locked(self, key: Any, value: Any, ttl: Optional[float], now: float) -> None:
        ttl = self.ttl if ttl is None else float(ttl)
        self._data[key] = _Entry(value=value, created_at=now, expires_at=now + ttl)

    # ---------- واجهة عامة ----------
    async def get(self, key: Any, allow_stale: bool = False) -> Tuple[bool, Any]:
        """يرجع (موجود, القيمة)."""
        if not self.enabled:
            return False, None
        now = time.monotonic()
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False, None
            if entry.expires_at > now:
                self._hits += 1
                return True, entry.value
            if allow_stale:
                return True, entry.value
            self._data.pop(key, None)
            return False, None

    async def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        async with self._lock:
            self._store_locked(key, value, ttl, now)
            self._purge_locked(now)

    async def delete(self, key: Any) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    async def get_or_set(self, key: Any, factory: Callable[[], Awaitable[Any]],
                         ttl: Optional[float] = None) -> Tuple[Any, bool]:
        """
        يجلب القيمة من الكاش، وإذا غائبة (أو منتهية) ينفّذ `factory` مرة واحدة فقط
        حتى لو نادتها عدة طلبات متزامنة (single-flight).

        يرجع (القيمة, from_cache).
        عند فشل `factory` ووجود قيمة منتهية صلاحيتها، تُعاد القيمة القديمة
        (stale) بدل رمي الخطأ إن كان serve_stale_on_error مفعّلاً.
        """
        if not self.enabled:
            value = await factory()
            return value, False

        now = time.monotonic()
        async with self._lock:
            entry = self._data.get(key)
            if entry is not None and entry.expires_at > now:
                self._hits += 1
                return entry.value, True
            stale_value = entry.value if entry is not None else None
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.get_event_loop().create_future()
                self._inflight[key] = future
                owner = True
            else:
                owner = False

        if not owner:
            # طلب آخر يجلب نفس المفتاح: ننتظر نتيجته بدل تكرار النداء الخارجي
            self._joined += 1
            value = await asyncio.shield(future)
            return value, False

        try:
            value = await factory()
        except Exception:
            self._errors += 1
            async with self._lock:
                self._inflight.pop(key, None)
            if not future.done():
                future.set_exception(_FactoryFailed())
            if self.serve_stale_on_error and stale_value is not None:
                return stale_value, True
            raise
        else:
            now = time.monotonic()
            async with self._lock:
                self._inflight.pop(key, None)
                self._misses += 1
                self._store_locked(key, value, ttl, now)
                self._purge_locked(now)
            if not future.done():
                future.set_result(value)
            return value, False

    def stats(self) -> Dict[str, Any]:
        now = time.monotonic()
        live = sum(1 for e in self._data.values() if e.expires_at > now)
        total = self._hits + self._misses
        return {
            "name": self.name,
            "enabled": self.enabled,
            "entries": len(self._data),
            "live_entries": live,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / total, 3) if total else 0.0,
            "errors": self._errors,
            "joined_inflight": self._joined,
            "ttl_seconds": self.ttl,
        }


class _FactoryFailed(Exception):
    """استثناء داخلي يُبلّغ الطلبات المنتظرة أن النداء الأصلي فشل."""
