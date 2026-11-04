from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normal(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).casefold()


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{1,35}", text.casefold()))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(part[i:i + 2] for i in range(len(part) - 1))
    return words - {"什么", "怎么", "知道", "一下", "这个", "那个", "帮我", "请问", "有没有", "是不是"}


def public_url(url: str) -> str:
    try:
        p = urlsplit(str(url).strip())
        host = (p.hostname or "").lower().rstrip(".")
        port = p.port
        if p.scheme not in {"http", "https"} or not host or p.username or p.password:
            raise ValueError("只接受公开网页地址")
        if port not in (None, 80, 443) or any(c.isspace() for c in url):
            raise ValueError("网页地址格式不符合要求")
        if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".lan")):
            raise ValueError("不读取本地或内网地址")
        try:
            if not ipaddress.ip_address(host).is_global:
                raise ValueError("不读取内网地址")
        except ValueError as exc:
            if "不读取" in str(exc):
                raise
            if "." not in host or re.fullmatch(r"[\d.]+", host):
                raise ValueError("只接受公开网站域名") from None
        netloc = f"[{host}]" if ":" in host else host
        if port and port not in (80, 443):
            netloc += ":" + str(port)
        query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"spm", "fbclid", "gclid"}]
        return urlunsplit((p.scheme, netloc, p.path or "/", urlencode(query), ""))
    except (TypeError, AttributeError):
        raise ValueError("网页地址无效") from None


def public_query(query: str, private_terms=()) -> str:
    query = re.sub(r"\s+", " ", str(query)).strip()
    if not 2 <= len(query) <= 128:
        raise ValueError("请把公开主题概括为2～128个字")
    if re.search(r"(?<!\d)\d{6,}(?!\d)|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|sk-[a-zA-Z0-9]{12,}|Bearer\s+\S+|system_reminder|FriendMessage|GroupMessage|\[At:", query, re.I):
        raise ValueError("搜索词含账号、联系方式或聊天标记，请改成不含个人信息的公开主题")
    if any(x and len(str(x)) >= 2 and str(x) in query for x in private_terms):
        raise ValueError("搜索词含当前群友的身份信息，请只保留公开主题")
    if re.search(r"(群友|我朋友|我同事|我家|主人|私聊|聊天记录).{0,12}(是谁|地址|电话|住|说过|喜欢|隐私)", query):
        raise ValueError("群友私事应查询原记忆，不发送到公开搜索")
    return query


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


class KnowledgeStore:
    def __init__(self, path: Path, now=time.time):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.now = now
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge (
          id INTEGER PRIMARY KEY, title TEXT NOT NULL, topic TEXT NOT NULL,
          version TEXT NOT NULL, summary TEXT NOT NULL, evidence TEXT NOT NULL,
          content_hash TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
          expires REAL NOT NULL, embedding TEXT NOT NULL DEFAULT '[]');
        CREATE INDEX IF NOT EXISTS knowledge_expiry ON knowledge(expires);
        CREATE TABLE IF NOT EXISTS revisions (
          id INTEGER PRIMARY KEY, knowledge_id INTEGER NOT NULL,
          snapshot TEXT NOT NULL, archived REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS daily_usage (
          day TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL,
          PRIMARY KEY(day,kind));
        CREATE TABLE IF NOT EXISTS feedback (
          owner TEXT NOT NULL, category TEXT NOT NULL, preference TEXT NOT NULL,
          quote TEXT NOT NULL, updated REAL NOT NULL,
          PRIMARY KEY(owner,category));
        """)
        self.db.commit()

    def day(self):
        return datetime.fromtimestamp(self.now(), timezone(timedelta(hours=8))).date().isoformat()

    def reserve(self, kind: str, limit: int) -> bool:
        if limit <= 0:
            return False
        with self.lock, self.db:
            cursor = self.db.execute("""INSERT INTO daily_usage VALUES(?,?,1)
                ON CONFLICT(day,kind) DO UPDATE SET count=count+1 WHERE daily_usage.count<?""",
                (self.day(), kind, int(limit)))
            return cursor.rowcount == 1

    def stats(self):
        with self.lock:
            total, active = self.db.execute("SELECT count(*),coalesce(sum(expires>?),0) FROM knowledge", (self.now(),)).fetchone()
            used = dict(self.db.execute("SELECT kind,count FROM daily_usage WHERE day=?", (self.day(),)))
            return {"total": total, "active": active, "expired": total - active, "usage": used}

    def save(self, title, topic, version, summary, evidence, ttl_days, embedding=None, max_records=1000):
        title, topic, version, summary = (str(x).strip() for x in (title, topic, version, summary))
        if not title or not topic or not 30 <= len(summary) <= 1800:
            raise ValueError("标题、主题不能为空，知识摘要需为30～1800字")
        digest = hashlib.sha256(normal(summary).encode()).hexdigest()
        now = self.now()
        with self.lock, self.db:
            old = self.db.execute("SELECT * FROM knowledge WHERE content_hash=? AND version=?", (digest, version)).fetchone()
            if old:
                self.db.execute("UPDATE knowledge SET evidence=?,updated=?,expires=? WHERE id=?",
                                (json.dumps(evidence, ensure_ascii=False), now, now + ttl_days * 86400, old['id']))
                return {'id': old['id'], 'updated_existing': True, 'duplicate_skipped': True,
                        'expires': now + ttl_days * 86400}
            if not old:
                rows = self.db.execute("SELECT * FROM knowledge WHERE version=? ORDER BY updated DESC", (version,)).fetchall()
                old = next((r for r in rows if normal(r['topic']) == normal(topic) or
                            SequenceMatcher(None, normal(r['title']), normal(title)).ratio() >= 0.9), None)
            if old:
                self.db.execute("INSERT INTO revisions(knowledge_id,snapshot,archived) VALUES(?,?,?)",
                                (old['id'], json.dumps(dict(old), ensure_ascii=False), now))
                record_id = old['id']
                self.db.execute("""UPDATE knowledge SET title=?,topic=?,version=?,summary=?,evidence=?,
                    content_hash=?,updated=?,expires=?,embedding=? WHERE id=?""",
                    (title[:120], topic[:120], version[:80], summary, json.dumps(evidence, ensure_ascii=False),
                     digest, now, now + ttl_days * 86400, json.dumps(embedding or []), record_id))
            else:
                if self.db.execute("SELECT count(*) FROM knowledge").fetchone()[0] >= max_records:
                    raise ValueError("知识数量达到上限，请先在管理中调整容量或整理旧资料")
                cursor = self.db.execute("""INSERT INTO knowledge(title,topic,version,summary,evidence,content_hash,
                    created,updated,expires,embedding) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (title[:120], topic[:120], version[:80], summary, json.dumps(evidence, ensure_ascii=False),
                     digest, now, now, now + ttl_days * 86400, json.dumps(embedding or [])))
                record_id = cursor.lastrowid
            return {"id": record_id, "updated_existing": bool(old), "expires": now + ttl_days * 86400}

    def search(self, query, vector=None, limit=3):
        keys = terms(query)
        if not keys:
            return []
        with self.lock:
            rows = self.db.execute("SELECT * FROM knowledge WHERE expires>? ORDER BY updated DESC LIMIT 1000", (self.now(),)).fetchall()
        ranked = []
        for index, row in enumerate(rows):
            title_keys = terms(row['title'] + ' ' + row['topic'] + ' ' + row['version'])
            body_keys = terms(row['summary'])
            matched = keys & (title_keys | body_keys)
            lexical = len(matched) / max(1, min(6, len(keys)))
            title_score = len(keys & title_keys) / max(1, min(4, len(keys)))
            semantic = cosine(vector, json.loads(row['embedding'])) if vector and index < 200 else 0
            if not ((matched and (len(matched) >= 2 or len(keys) <= 2) and lexical >= 0.3) or semantic >= 0.72):
                continue
            item = dict(row)
            item.pop('embedding')
            item['evidence'] = json.loads(item['evidence'])
            item['score'] = round(lexical + title_score + max(0, semantic - 0.5), 4)
            ranked.append(item)
        return sorted(ranked, key=lambda x: x['score'], reverse=True)[:limit]

    def save_feedback(self, owner, category, preference, quote, current_message):
        if category not in {'length', 'questions', 'repetition', 'humor', 'formality', 'tone'}:
            raise ValueError("交流偏好类别不支持")
        if not 4 <= len(normal(quote)) <= 400 or normal(quote) not in normal(current_message):
            raise ValueError("反馈必须有当前用户消息中的原话依据")
        if not 2 <= len(preference.strip()) <= 160:
            raise ValueError("请用2～160字概括交流偏好")
        with self.lock, self.db:
            self.db.execute("""INSERT INTO feedback VALUES(?,?,?,?,?)
                ON CONFLICT(owner,category) DO UPDATE SET preference=excluded.preference,
                quote=excluded.quote,updated=excluded.updated""", (owner, category, preference.strip(), quote, self.now()))

    def feedback(self, owner):
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT preference FROM feedback WHERE owner=? AND updated>? ORDER BY updated DESC LIMIT 3",
                                                  (owner, self.now() - 90 * 86400))]

    def close(self):
        with self.lock:
            self.db.close()
