"""Disk-backed whole-corpus deduplication and split assignment."""
from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import sqlite3

from .core import CurationConfig, NearIndex, choose_split, digest, jaccard, shingles, write_jsonl


MAX_BAND_DOCUMENTS = 2_000
MAX_CANDIDATE_PAIRS = 10_000_000


def _rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


class CorpusIndex:
    """External-memory merge of checksummed worker outputs in input order."""

    def __init__(self, path: Path, config: CurationConfig):
        self.config = config
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE document (
                seq INTEGER PRIMARY KEY, input_index INTEGER NOT NULL,
                id TEXT NOT NULL, sha TEXT NOT NULL, family TEXT NOT NULL,
                body TEXT NOT NULL, tokens TEXT, reason TEXT, split TEXT, group_sha TEXT
            );
            CREATE INDEX document_split ON document (split, seq);
            CREATE TABLE first_id (id TEXT PRIMARY KEY);
            CREATE TABLE exact (partition TEXT, sha TEXT, seq INTEGER,
                                PRIMARY KEY(partition, sha));
            CREATE TABLE band (band INTEGER, key TEXT, seq INTEGER,
                               PRIMARY KEY (band, key, seq));
            CREATE TABLE checked_pair (a INTEGER, b INTEGER, PRIMARY KEY(a, b));
            CREATE TABLE document_parent (node INTEGER PRIMARY KEY, parent INTEGER);
            CREATE TABLE family_parent (node TEXT PRIMARY KEY, parent TEXT);
            CREATE TABLE forbidden (kind TEXT, hash TEXT, PRIMARY KEY(kind, hash));
            CREATE TABLE alias (id_hash TEXT, sha TEXT, PRIMARY KEY(id_hash, sha));
        """)

    def __enter__(self) -> CorpusIndex:
        return self

    def __exit__(self, *_exc) -> None:
        self.db.close()

    def _root(self, table: str, node):
        parents = []
        parent, = self.db.execute(f"SELECT parent FROM {table} WHERE node=?", (node,)).fetchone()
        while parent != node:
            parents.append(node)
            node = parent
            parent, = self.db.execute(f"SELECT parent FROM {table} WHERE node=?", (node,)).fetchone()
        self.db.executemany(f"UPDATE {table} SET parent=? WHERE node=?",
                            ((node, child) for child in parents))
        return node

    def _join(self, table: str, left, right) -> None:
        a, b = sorted((self._root(table, left), self._root(table, right)))
        if a != b:
            self.db.execute(f"UPDATE {table} SET parent=? WHERE node=?", (a, b))

    def ingest(self, task_directories: list[Path], input_hashes: list[str]) -> None:
        for directory in task_directories:
            for row in _rows(directory / "suppression.jsonl"):
                self.db.executemany(
                    "INSERT OR IGNORE INTO forbidden VALUES (?, ?)", row.items()
                )
        if self.db.execute(
            "SELECT 1 FROM forbidden WHERE kind='doc_id_sha256' LIMIT 1"
        ).fetchone():
            for directory in task_directories:
                for row in _rows(directory / "candidates.jsonl"):
                    self.db.execute("""
                        INSERT OR IGNORE INTO forbidden
                        SELECT 'sha256', ? FROM forbidden
                        WHERE kind='doc_id_sha256' AND hash=?
                    """, (row["sha256"], digest(row["id"].encode())))
        hasher = NearIndex(Path(":memory:"), self.config.seed)
        try:
            for input_index, directory in enumerate(task_directories):
                for row in _rows(directory / "candidates.jsonl"):
                    self._add(row, input_index, input_hashes[input_index], hasher)
                self.db.commit()
        finally:
            hasher.close()

    def _add(self, row: dict, input_index: int, input_hash: str, hasher: NearIndex) -> None:
        sha, doc_id = row["sha256"], row["id"]
        id_hash = digest(doc_id.encode())
        family = digest(json.dumps(
            [row["source_id"], row["family_id"]], ensure_ascii=False, separators=(",", ":")
        ).encode())
        self.db.execute("INSERT OR IGNORE INTO alias VALUES (?, ?)", (id_hash, sha))
        row["input_sha256"] = input_hash
        row["input_index"] = input_index
        blocked = self.db.execute(
            "SELECT 1 FROM forbidden WHERE (kind='doc_id_sha256' AND hash=?) "
            "OR (kind='sha256' AND hash=?)", (id_hash, sha)
        ).fetchone()
        reason = "record_opt_out" if blocked else None
        if reason is None and not self.db.execute(
                "INSERT OR IGNORE INTO first_id VALUES (?)", (doc_id,)).rowcount:
            reason = "duplicate_document_id"
        seq = self.db.execute(
            "INSERT INTO document(input_index,id,sha,family,body,reason) VALUES (?,?,?,?,?,?)",
            (input_index, doc_id, sha, family, json.dumps(row, ensure_ascii=False), reason)
        ).lastrowid
        if reason:
            return
        self.db.execute("INSERT INTO document_parent VALUES (?, ?)", (seq, seq))
        self.db.execute("INSERT OR IGNORE INTO family_parent VALUES (?, ?)", (family, family))
        partition = sha[:2]
        if not self.db.execute(
            "INSERT OR IGNORE INTO exact VALUES (?,?,?)", (partition, sha, seq)
        ).rowcount:
            earlier, = self.db.execute(
                "SELECT seq FROM exact WHERE partition=? AND sha=?", (partition, sha)
            ).fetchone()
            self._join("document_parent", earlier, seq)
            self.db.execute("UPDATE document SET reason='duplicate_exact' WHERE seq=?", (seq,))
            return
        items = shingles(row["text"])
        if items:
            self.db.execute("UPDATE document SET tokens=? WHERE seq=?",
                            (json.dumps(sorted(items), ensure_ascii=False), seq))
            self.db.executemany("INSERT INTO band VALUES (?,?,?)",
                                ((band, key, seq) for band, key in hasher.bands(hasher.signature(items))))

    def assign(self) -> None:
        overloaded = self.db.execute(
            "SELECT 1 FROM band GROUP BY band,key HAVING count(*) > ? LIMIT 1",
            (MAX_BAND_DOCUMENTS,)
        ).fetchone()
        if overloaded:
            raise ValueError("deduplication band exceeds the safe candidate limit")
        pairs = self.db.execute("""
            SELECT left.seq, right.seq FROM band AS left
            JOIN band AS right ON right.band=left.band AND right.key=left.key AND right.seq>left.seq
            ORDER BY left.band,left.key,left.seq,right.seq
        """)
        count = 0
        for a, b in pairs:
            if not self.db.execute(
                "INSERT OR IGNORE INTO checked_pair VALUES (?,?)", (a, b)
            ).rowcount:
                continue
            count += 1
            if count > MAX_CANDIDATE_PAIRS:
                raise ValueError("deduplication exceeds the safe candidate pair limit")
            if self._root("document_parent", a) == self._root("document_parent", b):
                continue
            left, right = (frozenset(json.loads(raw)) for raw, in self.db.execute(
                "SELECT tokens FROM document WHERE seq IN (?,?) ORDER BY seq", (a, b)
            ))
            if jaccard(left, right) >= self.config.min_near_jaccard:
                self._join("document_parent", a, b)
        for seq, family, reason in self.db.execute(
            "SELECT seq,family,reason FROM document JOIN document_parent ON seq=node ORDER BY seq"
        ):
            representative = self._root("document_parent", seq)
            if representative != seq:
                first_family, = self.db.execute(
                    "SELECT family FROM document WHERE seq=?", (representative,)
                ).fetchone()
                self._join("family_parent", family, first_family)
                if reason is None:
                    self.db.execute("UPDATE document SET reason='duplicate_near' WHERE seq=?", (seq,))
        for seq, family in self.db.execute("SELECT seq,family FROM document WHERE reason IS NULL"):
            group = self._root("family_parent", family)
            self.db.execute("UPDATE document SET split=?,group_sha=? WHERE seq=?",
                            (choose_split(group, self.config.seed), group, seq))
        self.db.commit()

    def rows(self, split: str) -> Iterator[dict]:
        for body, group in self.db.execute(
            "SELECT body,group_sha FROM document WHERE split=? ORDER BY seq", (split,)
        ):
            yield {**json.loads(body), "split": split, "group_sha256": group}

    def write_audit(self, audit, directories: list[Path], aliases: Path) -> None:
        for input_index, directory in enumerate(directories):
            for row in _rows(directory / "audit.jsonl"):
                write_jsonl(audit, {**row, "input_index": input_index})
        for body, reason in self.db.execute("SELECT body,reason FROM document WHERE reason IS NOT NULL"):
            row = json.loads(body)
            write_jsonl(audit, {
                "input_index": row["input_index"], "line": row["line"],
                "input_line_sha256": row["input_line_sha256"], "source_id": row["source_id"],
                "decision": "rejected", "reason": reason,
            })
        with aliases.open("w", encoding="utf-8") as stream:
            for id_hash, sha in self.db.execute("SELECT id_hash,sha FROM alias ORDER BY id_hash,sha"):
                write_jsonl(stream, {"doc_id_sha256": id_hash, "sha256": sha})
