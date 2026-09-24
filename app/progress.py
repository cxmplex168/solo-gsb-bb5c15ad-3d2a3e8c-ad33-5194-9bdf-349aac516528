"""分片进度增量同步：事件流 + 一致性快照。

多端协作上传同一会话时，每个客户端都需要追踪**其他客户端**已提交的
区间。本模块提供两件套：

- **事件流**（``chunk_events`` 表）：普通 PUT 分片与服务端 copy-chunk
  成功登记时，在同一登记事务内追加一条同会话单调递增序号（``seq``）
  的事件；失败、重试冲突与分片修复（repair）不产生事件。
- **快照 + 游标**：``snapshot`` 在同一次 SQLite 读事务中返回当前已收
  区间与事件游标（当前最大 ``seq``）；``events_after`` 按游标分页
  返回其后的成功接收区间。

客户端先取快照再以游标追赶增量，两次请求之间的提交不会漏掉：
快照的游标与已收区间来自同一数据库快照，此后提交的区间必然带有
更大的事件序号，必然出现在后续事件页中。旧会话（迁移前已存在）
没有历史事件：其既有区间即快照基线，后续事件从该基线继续编号。
"""
from __future__ import annotations

import sqlite3

from . import db

# 事件页大小上限（与接口层 Query 校验保持一致）。
MAX_PAGE_SIZE = 100


def record_chunk_event(
    tx: sqlite3.Connection,
    session_id: str,
    start: int,
    end: int,
    received_at: str,
) -> int:
    """在分片登记事务内追加一条进度事件，返回分配到的会话内序号。

    必须在登记 chunks 行的同一个 BEGIN IMMEDIATE 事务内调用：写事务
    串行化保证 ``MAX(seq) + 1`` 不会并发重复；事务回滚（失败、重试
    冲突）时事件一并消失，因此持久化的事件流恰好对应成功登记的分片。
    """
    row = tx.execute(
        "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM chunk_events "
        "WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    seq = row["max_seq"] + 1
    tx.execute(
        "INSERT INTO chunk_events "
        "(session_id, seq, start_offset, end_offset, received_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, seq, start, end, received_at),
    )
    return seq


def snapshot(session_id: str) -> tuple[list[tuple[int, int]], int]:
    """在**同一次数据库读事务**中返回 (已收区间列表, 事件游标)。

    已收区间按 chunks 登记原样返回（未合并，由调用方决定是否合并）；
    事件游标为当前最大事件序号（无事件时为 0）。两个结果取自同一
    数据库快照：游标之后提交的区间必然能以事件形式被客户端追到。
    """
    conn = db.get_conn()
    conn.execute("BEGIN")
    try:
        received = [
            (r["start_offset"], r["end_offset"])
            for r in conn.execute(
                "SELECT start_offset, end_offset FROM chunks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        ]
        cursor = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS cursor FROM chunk_events "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()["cursor"]
    finally:
        # 纯读事务：回滚与提交效果相同，这里统一回滚释放读锁。
        conn.rollback()
    return received, cursor


def events_after(
    session_id: str, after: int, limit: int
) -> tuple[list[sqlite3.Row], bool]:
    """按序号升序返回游标 ``after`` 之后的事件，最多 ``limit`` 条。

    返回 (事件行列表, 是否还有更多)。单条 SELECT 自身就是一致快照，
    多取一条用于判断后续页，不额外开事务。
    """
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT seq, start_offset, end_offset, received_at FROM chunk_events "
        "WHERE session_id = ? AND seq > ? ORDER BY seq LIMIT ?",
        (session_id, after, limit + 1),
    ).fetchall()
    has_more = len(rows) > limit
    return rows[:limit], has_more
