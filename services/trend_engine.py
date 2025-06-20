# trend_manager_pg.py
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import settings    


# ───────────────────────────────────────────────
#  Connection helper
# ───────────────────────────────────────────────
def _get_conn():
    """
    Open a PostgreSQL connection using the credentials
    provided by `settings.py`.
    """
    return psycopg2.connect(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        dbname=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD,
        cursor_factory=RealDictCursor,
    )


# ───────────────────────────────────────────────
#  TrendManager
# ───────────────────────────────────────────────
class TrendManager:
    def __init__(self, cfg=settings, auto_init: bool = True):
        """
        cfg: a Settings instance (default = global `settings`).
             Makes unit-testing easy: just pass a different Settings object.
        """
        self.cfg = cfg
        if auto_init:
            self.init_database()

    # ---------- Schema bootstrap ----------
    def init_database(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS GeneratedContent (
            id                   BIGINT PRIMARY KEY,
            user_id              BIGINT,
            title                VARCHAR(255),
            prompt               TEXT,
            tags                 TEXT[],
            generated_image_url  VARCHAR(255),
            engagement_score     FLOAT,
            creation_timestamp   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    # ---------- Ingest ----------
    def add_content(
        self,
        *,
        id: int,
        user_id: int,
        title: str,
        prompt: str,
        tags: List[str],
        image_url: str,
        engagement_score: float = 0.0,
        created_at: Optional[datetime] = None,
    ) -> None:
        sql = """
        INSERT INTO GeneratedContent
              (id, user_id, title, prompt, tags, generated_image_url,
               engagement_score, creation_timestamp)
        VALUES (%s,  %s,      %s,    %s,     %s,   %s,                 %s,  %s)
        ON CONFLICT (id) DO NOTHING;
        """
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    id,
                    user_id,
                    title,
                    prompt,
                    tags,
                    image_url,
                    engagement_score,
                    created_at or datetime.now(timezone.utc),
                ),
            )
            conn.commit()

    # ---------- Retrieval ----------
    def fetch_top_performing_content(
        self, hours_back: int = 3, limit: int = 50
    ) -> List[Dict]:
        time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        sql = """
        SELECT id, user_id, title, prompt, tags,
               generated_image_url, engagement_score, creation_timestamp
        FROM   GeneratedContent
        WHERE  creation_timestamp >= %s
        ORDER  BY engagement_score DESC
        LIMIT  %s;
        """
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (time_threshold, limit))
            return cur.fetchall()         

    # ---------- Soft-max & Nucleus Sampling ----------
    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        # A higher temperature flattens the distribution
        if temperature <= 0:
            temperature = 1.0
        exps = np.exp((scores - scores.max()) / temperature)
        return exps / exps.sum()

    def _nucleus_sample(
        self,
        rows: List[Dict],
        p: float = 0.7,
        k: int = 5,
        temperature: float = 1.0
    ) -> List[Dict]:
        if not rows:
            return []
        scores = np.array([r["engagement_score"] for r in rows], float)
        probs = self._softmax(scores, temperature=temperature)
        order = probs.argsort()[::-1]
        cum = np.cumsum(probs[order])
        nucleus_cut = np.searchsorted(cum, p) + 1
        nucleus_idx = order[:nucleus_cut]
        nucleus_probs = probs[nucleus_idx] / probs[nucleus_idx].sum()
        chosen = np.random.choice(
            nucleus_idx, size=min(k, nucleus_cut), replace=False, p=nucleus_probs
        )
        return [rows[i] for i in chosen]
 
     # ─── Public API ─────────────────────────────────────────────────────
    def get_trending_prompts(
        self,
        hours_back: int = 3,
        top_n: int = 50,
        p_threshold: float = 0.5,
        num_samples: int = 5,
        temperature: float = 1.0
    ) -> List[Dict]:
        rows = self.fetch_top_performing_content(hours_back, top_n)
        return self._nucleus_sample(rows, p_threshold, num_samples, temperature=temperature)
