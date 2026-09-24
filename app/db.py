"""SQLite 访问层。

每个工作线程持有自己的连接（thread-local），WAL 模式下读写互不阻塞；
所有写操作统一通过 :func:`immediate` 以 BEGIN IMMEDIATE 开启事务，
把冲突检查与插入/更新放进同一个串行化事务，杜绝并发分片导致的重叠区间。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from . import settings

_local = threading.local()


def _db_path() -> Path:
    url = settings.database_url
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///"):])
    # 内存库等其他形式：交给 sqlite3 直接处理。
    return Path(url)


def get_conn() -> sqlite3.Connection:
    """返回当前线程的连接（惰性创建）。"""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path),
            timeout=30.0,
            isolation_level=None,  # 手动事务
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


@contextmanager
def immediate() -> Iterator[sqlite3.Connection]:
    """开启 BEGIN IMMEDIATE 事务，提交/回滚由上下文自动处理。"""
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


@contextmanager
def read_only() -> Iterator[sqlite3.Connection]:
    """打开一个与线程连接缓存隔离的只读连接（uri mode=ro）。

    专供运维只读查询使用：连接本身被 PRAGMA query_only 再次兜底，
    既不会写库（不触碰 last_access_at / TTL），也不参与任何事务状态。
    每次调用独立建连、用完即关，绝不复用可写连接。
    """
    path = _db_path().resolve()
    uri = f"file:{quote(str(path))}?mode=ro"
    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 结构与迁移
#
# 使用 SQLite 的 PRAGMA user_version 记录结构版本，启动时按版本号增量迁移：
#   v1：upload_sessions / chunks（首个发布版本）
#   v2：download_tickets（受控下载票据）
#   v3：idempotency_keys（创建会话幂等键）
#   v4：chunk_claims（分片认领令牌，仅存令牌 SHA-256）
#   v5：chunk_events（分片进度事件流，同会话单调递增序号）
# 全部 DDL 均为 IF NOT EXISTS，重复执行安全；老库数据在迁移中保留。
# ---------------------------------------------------------------------------

MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS upload_sessions (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    total_size     INTEGER NOT NULL CHECK (total_size >= 0),
    sha256_expected TEXT NOT NULL,
    chunk_size     INTEGER NOT NULL CHECK (chunk_size >= 0),
    status         TEXT NOT NULL CHECK (status IN ('uploading','completed','aborted','expired')),
    bytes_received INTEGER NOT NULL DEFAULT 0 CHECK (bytes_received >= 0),
    created_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    last_access_at TEXT NOT NULL,
    completed_at   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES upload_sessions(id) ON DELETE CASCADE,
    start_offset  INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset    INTEGER NOT NULL CHECK (end_offset > start_offset),
    size          INTEGER NOT NULL CHECK (size > 0),
    received_at   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_session_offsets
    ON chunks(session_id, start_offset);
CREATE INDEX IF NOT EXISTS idx_chunks_session_interval
    ON chunks(session_id, start_offset, end_offset);
CREATE INDEX IF NOT EXISTS idx_sessions_expires
    ON upload_sessions(status, expires_at);
"""

MIGRATION_2 = """
-- 受控下载票据：只保存随机票据明文的 SHA-256，明文仅在签发响应中出现一次。
-- 剩余次数在 IMMEDIATE 事务内原子扣减（见 main._claim_ticket_download）。
CREATE TABLE IF NOT EXISTS download_tickets (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL
                            REFERENCES upload_sessions(id) ON DELETE CASCADE,
    token_hash          TEXT NOT NULL UNIQUE,
    max_downloads       INTEGER NOT NULL CHECK (max_downloads BETWEEN 1 AND 100),
    remaining_downloads INTEGER NOT NULL
                            CHECK (remaining_downloads >= 0
                                   AND remaining_downloads <= max_downloads),
    revoked             INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),
    created_at          TEXT NOT NULL,
    expires_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_session
    ON download_tickets(session_id);
CREATE INDEX IF NOT EXISTS idx_tickets_expires
    ON download_tickets(revoked, expires_at);
"""

MIGRATION_3 = """
-- 创建会话幂等键：只保存键明文的 SHA-256 与规范化请求指纹，明文绝不落库。
-- 键记录与会话在同一事务写入；会话删除时经 ON DELETE CASCADE 一并删除。
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key_hash            TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL
                            REFERENCES upload_sessions(id) ON DELETE CASCADE,
    request_fingerprint TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_idempotency_keys_session
    ON idempotency_keys(session_id);
"""

MIGRATION_4 = """
-- 分片认领：上传中会话对“尚未接收”的半开区间建立的限时令牌。
-- 只保存随机令牌明文的 SHA-256，明文仅在 POST /chunk-claims 响应中出现一次。
-- 相交的有效（未过期）认领最多一个（创建时在 IMMEDIATE 事务内复核，见
-- main.create_chunk_claim）；分片成功提交即在同一事务内消费（删除）认领行。
-- 会话删除时经 ON DELETE CASCADE 一并删除；完成/过期时由主流程显式清理。
CREATE TABLE IF NOT EXISTS chunk_claims (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL
                        REFERENCES upload_sessions(id) ON DELETE CASCADE,
    start_offset  INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset    INTEGER NOT NULL CHECK (end_offset > start_offset),
    token_hash    TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_claims_session
    ON chunk_claims(session_id, start_offset, end_offset);
CREATE INDEX IF NOT EXISTS idx_chunk_claims_expires
    ON chunk_claims(expires_at);
"""

MIGRATION_5 = """
-- 分片进度事件流：每次分片成功登记（普通 PUT 与 copy-chunk）在同一事务内
-- 追加一条事件，seq 在同一会话内单调递增（登记事务内取 MAX(seq)+1，
-- BEGIN IMMEDIATE 串行化保证不重复；事务回滚则事件一并消失）。
-- 失败、重试冲突与分片修复（repair）不产生事件。
-- 旧会话迁移后本表为空：其既有 chunks 区间即快照基线，后续事件从 seq=1 继续。
-- 会话删除时经 ON DELETE CASCADE 一并删除。
CREATE TABLE IF NOT EXISTS chunk_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL
                        REFERENCES upload_sessions(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL CHECK (seq > 0),
    start_offset  INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset    INTEGER NOT NULL CHECK (end_offset > start_offset),
    received_at   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_events_session_seq
    ON chunk_events(session_id, seq);
"""

# 版本号 -> 迁移脚本；版本号同时写入 PRAGMA user_version。
_MIGRATIONS: dict[int, str] = {
    1: MIGRATION_1,
    2: MIGRATION_2,
    3: MIGRATION_3,
    4: MIGRATION_4,
    5: MIGRATION_5,
}
LATEST_USER_VERSION = max(_MIGRATIONS)


def init_db() -> None:
    """建库或把现有库增量迁移到最新结构（老数据保留）。"""
    conn = get_conn()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        # 全新库：依次执行全部迁移。
        steps = range(1, LATEST_USER_VERSION + 1)
    else:
        # 老库：只补齐缺失的版本。
        steps = range(version + 1, LATEST_USER_VERSION + 1)
    for step in steps:
        conn.executescript(_MIGRATIONS[step])
        # executescript 会提交事务；user_version 必须在其后单独写入。
        conn.execute(f"PRAGMA user_version = {step}")
