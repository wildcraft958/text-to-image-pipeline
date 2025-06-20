"""
utils/deconstruct_prompts.py
Phase-1 · Prompt Component Library Builder
──────────────────────────────────────────
• Fetches trending prompts via TrendManager (PostgreSQL)
• Uses the **new** Google Gen AI SDK (≥ 1.0). No more `genai.configure`.
• Writes component_library_YYYYMMDD_HHMM.json into <project-root>/samples/
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from google import genai                                
from google.genai import types
from pydantic import BaseModel, Field

from services.trend_engine import TrendManager
from config.settings import settings                    # unified config


# ───────────────────────────────
# Logging
# ───────────────────────────────
log = logging.getLogger("prompt-deconstructor")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────
# Pydantic schema for components
# ───────────────────────────────
class PromptComponents(BaseModel):
    subject: str = Field(description="Main subject")
    action_setting: str = Field(description="Action and/or setting")
    emotional_tone: str = Field(description="Mood / emotion")
    lighting: str = Field(description="Lighting description")
    color_palette: str = Field(description="Dominant colours")
    camera_angle: str = Field(description="Shot or angle")
    style_aesthetic: str = Field(description="Art style or aesthetic")
    composition_framing: str = Field(description="Composition / framing")
    original_prompt: str
    engagement_score: float


# ───────────────────────────────
# Gemini wrapper
# ───────────────────────────────
class GeminiClient:
    def __init__(self):
        if settings.GOOGLE_AI_STUDIO_API_KEY:
            self.client = genai.Client(api_key=settings.GOOGLE_AI_STUDIO_API_KEY)
            log.info("✅ GenAI client initialised (model %s)", settings.LLM_MODEL)
        else:
            self.client = None
            log.warning("⚠️  GOOGLE_AI_STUDIO_API_KEY not set – running offline")

    # ---------- prompt builder ----------
    @staticmethod
    def _build_prompt(prompt: str, score: float) -> str:
        return (
            "You are an expert prompt analyst. Break the given prompt into structured "
            "components and return JSON only, matching the provided schema.\n\n"
            f"PROMPT: {prompt}\n"
            f"ENGAGEMENT_SCORE: {score}"
        )

    # ---------- public ----------
    def deconstruct(self, prompt: str, score: float) -> Dict:
        """Return a dict matching PromptComponents."""
        if not self.client:
            # offline stub
            return PromptComponents(
                subject="Unknown",
                action_setting="Unknown",
                emotional_tone="Unknown",
                lighting="Unknown",
                color_palette="Unknown",
                camera_angle="Unknown",
                style_aesthetic="Unknown",
                composition_framing="Unknown",
                original_prompt=prompt,
                engagement_score=score,
            ).model_dump()

        try:
            cfg = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=400,
                top_p=0.7,
                response_mime_type="application/json",
                response_schema=PromptComponents.model_json_schema(),
            )

            resp = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=self._build_prompt(prompt, score),
                config=cfg,
            )
            return json.loads(resp.text)
        except Exception as exc:
            log.error("❌ GenAI error (%s) – returning fallback JSON", exc)
            return PromptComponents(
                subject="Unknown",
                action_setting="Unknown",
                emotional_tone="Unknown",
                lighting="Unknown",
                color_palette="Unknown",
                camera_angle="Unknown",
                style_aesthetic="Unknown",
                composition_framing="Unknown",
                original_prompt=prompt,
                engagement_score=score,
            ).model_dump()


# ───────────────────────────────
# PromptDeconstructor
# ───────────────────────────────
class PromptDeconstructor:
    def __init__(self):
        self.trend_manager = TrendManager()
        self.llm = GeminiClient()

    # ---------- soft-max + nucleus sampling ----------
    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        exps = np.exp((scores - scores.max()) / temperature)
        return exps / exps.sum()

    def _nucleus_sample(
        self, rows: List[Dict], p: float = 0.7, k: int = 5
    ) -> List[Dict]:
        if not rows:
            return []
        scores = np.array([r["engagement_score"] for r in rows], float)
        probs = self._softmax(scores)
        order = probs.argsort()[::-1]
        cum = np.cumsum(probs[order])
        nucleus_sz = np.searchsorted(cum, p) + 1
        nucleus_idx = order[:nucleus_sz]
        nucleus_probs = probs[nucleus_idx] / probs[nucleus_idx].sum()
        chosen = np.random.choice(
            nucleus_idx, size=min(k, nucleus_sz), replace=False, p=nucleus_probs
        )
        return [rows[i] for i in chosen]

    # ---------- main ----------
    def build_library(
        self,
        hours_back: int = 24,
        top_n: int = 100,
        p_threshold: float = 0.7,
        num_samples: int = 5,
    ) -> List[Dict]:
        log.info("🔍 Fetching trending prompts")
        rows = self.trend_manager.get_trending_prompts(
            hours_back=hours_back,
            top_n=top_n,
            p_threshold=p_threshold,
            num_samples=num_samples,
        )
        if not rows:
            log.warning("No trending prompts found")
            return []

        log.info("🧠 Deconstructing %d prompts", len(rows))
        components: List[Dict] = []
        for idx, row in enumerate(rows, 1):
            log.info("[%d/%d] score=%.1f  %.55s",
                     idx, len(rows), row["engagement_score"],
                     row["prompt"].replace("\n", " ")[:55])
            comp = self.llm.deconstruct(row["prompt"], row["engagement_score"])
            components.append(comp)

        return components

    # ---------- I/O ----------
    @staticmethod
    def save(data: List[Dict], path: Path):
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        log.info("💾 Saved component library → %s", path)


# ───────────────────────────────
# CLI runner
# ───────────────────────────────
def main():
    log.info("🚀 Prompt Component Library Builder")
    decon = PromptDeconstructor()
    library = decon.build_library(
        hours_back=24, top_n=100, p_threshold=0.7, num_samples=5
    )

    if library:
        out = SAMPLES_DIR / f"component_library_{datetime.now():%Y%m%d_%H%M}.json"
        decon.save(library, out)
        # simple stats
        subjects = {c["subject"] for c in library}
        log.info("📊 Unique subjects: %d", len(subjects))
    else:
        log.warning("No components extracted.")


if __name__ == "__main__":
    main()
