"""اختبارات وحدة الكاش (TTL + single-flight + القيم القديمة عند الفشل)."""

import asyncio

import pytest

from cache import TTLCache


@pytest.mark.asyncio
async def test_set_and_get_within_ttl():
    cache = TTLCache("t", ttl=30)
    await cache.set("a", 1)
    assert await cache.get("a") == (True, 1)


@pytest.mark.asyncio
async def test_expiry_after_ttl():
    cache = TTLCache("t", ttl=0.05)
    await cache.set("a", 1)
    await asyncio.sleep(0.08)
    assert await cache.get("a") == (False, None)


@pytest.mark.asyncio
async def test_get_or_set_caches_result_and_reports_source():
    cache = TTLCache("t", ttl=30)
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        return "value"

    value1, cached1 = await cache.get_or_set("k", factory)
    value2, cached2 = await cache.get_or_set("k", factory)
    assert value1 == value2 == "value"
    assert cached1 is False and cached2 is True
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_singleflight_runs_factory_once_for_concurrent_calls():
    cache = TTLCache("t", ttl=30)
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return "shared"

    results = await asyncio.gather(*(cache.get_or_set("k", factory) for _ in range(5)))
    assert calls["n"] == 1
    assert all(value == "shared" for value, _ in results)
    assert cache.stats()["joined_inflight"] == 4


@pytest.mark.asyncio
async def test_serves_stale_value_when_upstream_fails():
    cache = TTLCache("t", ttl=0.05, serve_stale_on_error=True)
    await cache.set("k", "قديمة")
    await asyncio.sleep(0.08)

    async def broken():
        raise RuntimeError("الخدمة الخارجية معطّلة")

    value, cached = await cache.get_or_set("k", broken)
    assert value == "قديمة" and cached is True


@pytest.mark.asyncio
async def test_raises_when_upstream_fails_without_stale():
    cache = TTLCache("t", ttl=0.05, serve_stale_on_error=False)

    async def broken():
        raise RuntimeError("فشل")

    with pytest.raises(RuntimeError):
        await cache.get_or_set("k", broken)
    assert cache.stats()["errors"] == 1


@pytest.mark.asyncio
async def test_disabled_cache_always_calls_factory():
    cache = TTLCache("t", ttl=30, enabled=False)
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        return calls["n"]

    assert await cache.get_or_set("k", factory) == (1, False)
    assert await cache.get_or_set("k", factory) == (2, False)
    assert await cache.get("k") == (False, None)


@pytest.mark.asyncio
async def test_max_entries_is_enforced():
    cache = TTLCache("t", ttl=30, max_entries=3)
    for i in range(10):
        await cache.set(f"k{i}", i)
    assert cache.stats()["entries"] <= 3


@pytest.mark.asyncio
async def test_stats_and_clear():
    cache = TTLCache("t", ttl=30)
    await cache.set("a", 1)
    await cache.get("a")
    await cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["live_entries"] == 1
    await cache.clear()
    assert cache.stats()["entries"] == 0
