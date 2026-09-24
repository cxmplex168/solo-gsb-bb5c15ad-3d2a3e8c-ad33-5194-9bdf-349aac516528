"""分片进度增量同步：一致性快照、事件流分页、不产生事件的路径、基线迁移。"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, progress, settings, storage
from app.cleanup import run_cleanup


def _create(client: TestClient, data: bytes, name: str = "data.bin") -> str:
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _put(client: TestClient, sid: str, start: int, end: int, content: bytes, **kw):
    headers = {}
    if "sha" in kw:
        headers["X-Chunk-SHA256"] = kw["sha"]
    return client.put(
        f"/api/v1/sessions/{sid}/chunks?start={start}&end={end}",
        content=content,
        headers=headers,
    )


def _snapshot(client: TestClient, sid: str):
    return client.get(f"/api/v1/sessions/{sid}/progress")


def _events(client: TestClient, sid: str, after: int = 0, limit: int | None = None):
    url = f"/api/v1/sessions/{sid}/progress/events?after={after}"
    if limit is not None:
        url += f"&limit={limit}"
    return client.get(url)


def _event_rows(sid: str) -> list[sqlite3.Row]:
    return db.get_conn().execute(
        "SELECT * FROM chunk_events WHERE session_id = ? ORDER BY seq", (sid,)
    ).fetchall()


def _make_completed(client: TestClient, data: bytes) -> str:
    """创建 + 单片 PUT + 完成，返回来源会话 id。"""
    sid = _create(client, data)
    r = _put(client, sid, 0, len(data), data)
    assert r.status_code == 200, r.text
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    return sid


# ---------------------------------------------------------------------------
# 快照：与游标取自同一次读事务
# ---------------------------------------------------------------------------


def test_snapshot_empty_session(client: TestClient) -> None:
    sid = _create(client, b"x" * 1000)
    r = _snapshot(client, sid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["status"] == "uploading"
    assert body["total_size"] == 1000
    assert body["bytes_received"] == 0
    assert body["received_ranges"] == []
    assert body["cursor"] == 0


def test_snapshot_merges_received_ranges_and_tracks_cursor(client: TestClient) -> None:
    data = bytes((i * 3 + 1) % 256 for i in range(500))
    sid = _create(client, data)

    # 乱序提交三片；[0,100) 与 [100,250) 相邻，快照中应合并。
    assert _put(client, sid, 100, 250, data[100:250]).status_code == 200
    assert _put(client, sid, 400, 500, data[400:500]).status_code == 200
    assert _put(client, sid, 0, 100, data[0:100]).status_code == 200

    body = _snapshot(client, sid).json()
    assert body["received_ranges"] == [
        {"start": 0, "end": 250},
        {"start": 400, "end": 500},
    ]
    assert body["bytes_received"] == 350
    # 三次成功登记 → 三个事件，游标为最大序号。
    assert body["cursor"] == 3


# ---------------------------------------------------------------------------
# 事件流：只有成功登记才产生事件
# ---------------------------------------------------------------------------


def test_events_stream_ordered_and_cursor_filtered(client: TestClient) -> None:
    data = bytes((i * 7 + 2) % 256 for i in range(300))
    sid = _create(client, data)
    for start, end in ((0, 100), (200, 300), (100, 200)):
        assert _put(client, sid, start, end, data[start:end]).status_code == 200

    r = _events(client, sid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_more"] is False
    assert body["next_cursor"] == 3
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    # 事件按登记顺序记录区间（不是按偏移排序）。
    assert [(e["start"], e["end"]) for e in body["events"]] == [
        (0, 100),
        (200, 300),
        (100, 200),
    ]
    for event in body["events"]:
        # received_at 为可解析的 UTC ISO-8601。
        storage.parse_iso(event["received_at"])

    # 游标过滤：after=1 只返回其后的两事件；空页游标保持不变。
    body = _events(client, sid, after=1).json()
    assert [e["seq"] for e in body["events"]] == [2, 3]
    assert body["next_cursor"] == 3
    body = _events(client, sid, after=3).json()
    assert body["events"] == []
    assert body["next_cursor"] == 3
    assert body["has_more"] is False


def test_failed_and_conflicting_puts_produce_no_events(client: TestClient) -> None:
    data = b"f" * 400
    sid = _create(client, data)
    assert _put(client, sid, 0, 100, data[:100]).status_code == 200
    assert _snapshot(client, sid).json()["cursor"] == 1

    # 重试冲突（与已收区间重叠）→ 409，无新事件。
    assert _put(client, sid, 50, 150, data[50:150]).status_code == 409
    # 逐片摘要不匹配 → 422，无新事件。
    bad_sha = hashlib.sha256(b"not-the-content").hexdigest()
    assert _put(client, sid, 100, 200, data[100:200], sha=bad_sha).status_code == 422
    # 长度与区间不符 → 400，无新事件。
    assert _put(client, sid, 100, 200, data[100:150]).status_code == 400
    # 越界 → 400，无新事件。
    assert _put(client, sid, 300, 500, data[:200]).status_code == 400

    assert _snapshot(client, sid).json()["cursor"] == 1
    assert len(_event_rows(sid)) == 1
    # 修正后重传成功：事件继续递增，不留空洞之外的伪事件。
    assert _put(client, sid, 100, 200, data[100:200]).status_code == 200
    assert [r["seq"] for r in _event_rows(sid)] == [1, 2]


def test_repair_produces_no_event(client: TestClient) -> None:
    data = bytes((i * 11 + 5) % 256 for i in range(200))
    sid = _create(client, data)
    assert _put(client, sid, 0, 100, data[:100]).status_code == 200
    assert _put(client, sid, 100, 200, data[100:]).status_code == 200
    assert _snapshot(client, sid).json()["cursor"] == 2

    # 原位修复已登记分片：成功但属于内容替换，不产生新事件。
    fixed = bytes((i * 13 + 7) % 256 for i in range(100))
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks/repair?start=0&end=100", content=fixed
    )
    assert r.status_code == 200, r.text

    snap = _snapshot(client, sid).json()
    assert snap["cursor"] == 2
    assert len(_event_rows(sid)) == 2
    # 已收区间不变。
    assert snap["received_ranges"] == [{"start": 0, "end": 200}]


def test_copy_chunk_shares_the_same_event_stream(client: TestClient) -> None:
    data = bytes((i * 5 + 3) % 256 for i in range(1000))
    src = _make_completed(client, data)
    tgt = _create(client, data, name="target.bin")

    assert _put(client, tgt, 0, 100, data[:100]).status_code == 200
    r = client.post(
        f"/api/v1/sessions/{tgt}/copy-chunk",
        json={
            "source_session_id": src,
            "source_start": 500,
            "source_end": 800,
            "target_start": 100,
        },
    )
    assert r.status_code == 200, r.text

    body = _events(client, tgt).json()
    assert [(e["seq"], e["start"], e["end"]) for e in body["events"]] == [
        (1, 0, 100),
        (2, 100, 400),
    ]
    assert _snapshot(client, tgt).json()["cursor"] == 2


# ---------------------------------------------------------------------------
# 快照 + 游标追赶：两次请求之间的提交不会漏掉
# ---------------------------------------------------------------------------


def test_snapshot_then_catch_up_misses_nothing(client: TestClient) -> None:
    data = bytes((i * 17 + 1) % 256 for i in range(600))
    sid = _create(client, data)

    # 客户端 A 已提交两片；客户端 B 此刻取快照建立本地视图。
    assert _put(client, sid, 0, 100, data[:100]).status_code == 200
    assert _put(client, sid, 100, 200, data[100:200]).status_code == 200
    snap = _snapshot(client, sid).json()
    view = {(r["start"], r["end"]) for r in snap["received_ranges"]}
    cursor = snap["cursor"]

    # 快照之后、追赶之前：其他客户端又提交了三片（含一片 copy-chunk 场景外的普通片）。
    assert _put(client, sid, 200, 300, data[200:300]).status_code == 200
    assert _put(client, sid, 400, 500, data[400:500]).status_code == 200
    assert _put(client, sid, 300, 400, data[300:400]).status_code == 200

    # B 用快照游标追增量：恰好拿到三次提交，不重不漏。
    body = _events(client, sid, after=cursor).json()
    assert [e["seq"] for e in body["events"]] == [3, 4, 5]
    for e in body["events"]:
        view.add((e["start"], e["end"]))

    # 合并后的本地视图与全新快照完全一致。
    merged = storage.merge_intervals(sorted(view))
    fresh = _snapshot(client, sid).json()
    assert merged == [(r["start"], r["end"]) for r in fresh["received_ranges"]]


# ---------------------------------------------------------------------------
# 分页
# ---------------------------------------------------------------------------


def test_events_pagination_walks_entire_stream(client: TestClient) -> None:
    data = bytes(range(256))[:105]
    sid = _create(client, data)
    for i in range(105):  # 105 个 1 字节分片 → 105 个事件
        assert _put(client, sid, i, i + 1, data[i : i + 1]).status_code == 200

    # 默认页大小 100：第一页 100 条且还有下一页。
    first = _events(client, sid).json()
    assert len(first["events"]) == 100
    assert first["has_more"] is True
    assert first["next_cursor"] == 100

    # 用小页翻完全部事件：seq 连续 1..105，最后一页 has_more=false。
    seen: list[int] = []
    cursor, limit = 0, 10
    while True:
        page = _events(client, sid, after=cursor, limit=limit).json()
        seen.extend(e["seq"] for e in page["events"])
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
    assert seen == list(range(1, 106))
    assert cursor == 105

    # 尾页之后继续轮询：空页，游标不动。
    tail = _events(client, sid, after=cursor, limit=limit).json()
    assert tail["events"] == [] and tail["next_cursor"] == 105


def test_events_page_size_and_cursor_validation(client: TestClient) -> None:
    sid = _create(client, b"y" * 10)
    assert _events(client, sid, limit=0).status_code == 422
    assert _events(client, sid, limit=101).status_code == 422
    assert _events(client, sid, after=-1).status_code == 422
    assert _events(client, sid, limit=1).status_code == 200
    assert _events(client, sid, limit=100).status_code == 200


# ---------------------------------------------------------------------------
# 会话生命周期
# ---------------------------------------------------------------------------


def test_progress_endpoints_session_lifecycle(client: TestClient) -> None:
    data = b"z" * 200
    sid = _create(client, data)
    assert _put(client, sid, 0, 200, data).status_code == 200

    # 未知会话：404。
    assert _snapshot(client, "nope").status_code == 404
    assert _events(client, "nope").status_code == 404

    # 完成后快照与事件仍可读（chunks 与事件行保留）。
    assert client.post(f"/api/v1/sessions/{sid}/complete").status_code == 200
    snap = _snapshot(client, sid).json()
    assert snap["status"] == "completed"
    assert snap["received_ranges"] == [{"start": 0, "end": 200}]
    assert snap["cursor"] == 1
    assert len(_events(client, sid).json()["events"]) == 1

    # 放弃会话：接口 404，事件行经外键级联删除。
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert _snapshot(client, sid).status_code == 404
    assert _events(client, sid).status_code == 404
    assert _event_rows(sid) == []


def test_progress_endpoints_expired_session(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ttl_seconds", 1)
    sid = _create(client, b"e" * 100)
    assert _put(client, sid, 0, 50, b"e" * 50).status_code == 200

    time.sleep(1.1)
    assert sid in run_cleanup()

    assert _snapshot(client, sid).status_code == 410
    assert _events(client, sid).status_code == 410


# ---------------------------------------------------------------------------
# 并发与事务性
# ---------------------------------------------------------------------------


def test_concurrent_puts_get_unique_monotonic_seqs(client: TestClient) -> None:
    data = b"c" * 800
    sid = _create(client, data)
    results: list[int] = []
    lock = threading.Lock()

    def upload(i: int) -> None:
        r = _put(client, sid, i * 100, i * 100 + 100, data[i * 100 : i * 100 + 100])
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [200] * 8
    # 会话锁 + IMMEDIATE 事务串行化：序号恰好是 1..8，无重复无缺口。
    assert [r["seq"] for r in _event_rows(sid)] == list(range(1, 9))
    assert _snapshot(client, sid).json()["cursor"] == 8


def test_event_and_chunk_registration_are_atomic(client: TestClient, monkeypatch) -> None:
    """事件写入失败时整个登记事务回滚：chunks 与事件都不残留。"""
    data = b"a" * 200
    sid = _create(client, data)

    class InjectedDBError(RuntimeError):
        pass

    def failing_record(*args, **kwargs):
        raise InjectedDBError("injected event insert failure")

    monkeypatch.setattr("app.main.progress.record_chunk_event", failing_record)
    with pytest.raises(InjectedDBError):
        _put(client, sid, 0, 100, data[:100])
    monkeypatch.undo()

    # 同事务回滚：分片未登记、无事件、已收字节为 0。
    assert _event_rows(sid) == []
    snap = _snapshot(client, sid).json()
    assert snap["received_ranges"] == []
    assert snap["bytes_received"] == 0
    assert snap["cursor"] == 0

    # 恢复后重试成功，事件从 1 开始。
    assert _put(client, sid, 0, 100, data[:100]).status_code == 200
    assert [r["seq"] for r in _event_rows(sid)] == [1]


# ---------------------------------------------------------------------------
# 老库自动迁移：既有区间即快照基线，事件从基线继续
# ---------------------------------------------------------------------------


def test_init_db_migrates_v4_database(monkeypatch, tmp_path: Path) -> None:
    """v4 老库（无 chunk_events）平滑升级到 v5：既有 chunks 成为快照基线。"""
    db_file = tmp_path / "v4.db"
    raw = sqlite3.connect(str(db_file))
    for step in (1, 2, 3, 4):
        raw.executescript(db._MIGRATIONS[step])
    raw.execute("PRAGMA user_version = 4")
    raw.execute(
        "INSERT INTO upload_sessions VALUES ("
        "'sid9','old.bin',100,'" + "c" * 64 + "',0,'uploading',60,"
        "'2026-01-01T00:00:00Z','2099-01-02T00:00:00Z','2026-01-01T00:00:00Z',NULL)"
    )
    # 迁移前已登记的两个分片：它们没有、也不应回填事件。
    raw.execute(
        "INSERT INTO chunks (session_id, start_offset, end_offset, size, received_at) "
        "VALUES ('sid9', 0, 30, 30, '2026-01-01T00:00:00Z')"
    )
    raw.execute(
        "INSERT INTO chunks (session_id, start_offset, end_offset, size, received_at) "
        "VALUES ('sid9', 30, 60, 30, '2026-01-01T00:00:00Z')"
    )
    raw.commit()
    raw.close()

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_file}")
    # 丢弃可能存在的线程缓存连接，强制按新 DATABASE_URL 建连。
    db._local.conn = None
    try:
        db.init_db()
        conn = db.get_conn()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.LATEST_USER_VERSION

        # 基线：既有区间直接在快照中返回，游标为 0（无历史事件）。
        received, cursor = progress.snapshot("sid9")
        assert sorted(received) == [(0, 30), (30, 60)]
        assert cursor == 0
        assert progress.events_after("sid9", 0, 100) == ([], False)

        # 迁移后的新登记从基线继续编号：第一个事件 seq=1。
        with db.immediate() as tx:
            tx.execute(
                "INSERT INTO chunks "
                "(session_id, start_offset, end_offset, size, received_at) "
                "VALUES ('sid9', 60, 100, 40, '2026-01-01T00:10:00Z')"
            )
            seq = progress.record_chunk_event(tx, "sid9", 60, 100, "2026-01-01T00:10:00Z")
        assert seq == 1

        rows, has_more = progress.events_after("sid9", 0, 100)
        assert has_more is False
        assert [(r["seq"], r["start_offset"], r["end_offset"]) for r in rows] == [
            (1, 60, 100)
        ]
        # 快照随之推进：基线 + 新事件区间，游标走到 1。
        received, cursor = progress.snapshot("sid9")
        assert sorted(received) == [(0, 30), (30, 60), (60, 100)]
        assert cursor == 1

        # 再次初始化幂等。
        db.init_db()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.LATEST_USER_VERSION
    finally:
        db._local.conn = None
