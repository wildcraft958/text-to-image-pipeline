# samples/prompt_deconstructor.py
"""
Builds a component-library from the most-engaging prompts
stored in the GeneratedContent table.

Steps
1. fetch trending rows through TrendManager              (PostgreSQL)
2. call Gemini (Google GenAI) to deconstruct each prompt
3. save & analyse the structured output
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Type

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import settings                  
from services.trend_engine import TrendManager   

# ───────────────────────────────
# folders & logging
# ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # one level up from utils/
SAMPLES_DIR  = PROJECT_ROOT / "samples"                   # …/samples
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)          

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("prompt-deconstructor")



# ───────────────────────────────
# Pydantic model for LLM output
# ───────────────────────────────
class PromptComponents(BaseModel):
    subject: str = Field(description="Main subject of the image.")
    action_setting: str = Field(description="What the subject is doing / where.")
    emotional_tone: str = Field(description="Overall mood.")
    lighting: str = Field(description="Lighting conditions & effects.")
    color_palette: str = Field(description="Dominant colours / palette.")
    camera_angle: str = Field(description="Perspective or shot type.")
    style_aesthetic: str = Field(description="Artistic / visual style.")
    composition_framing: str = Field(description="Composition & framing.")
    original_prompt: str
    engagement_score: float


# ───────────────────────────────
# Gemini wrapper
# ───────────────────────────────
class GeminiClient:
    def __init__(self):
        if settings.GOOGLE_AI_STUDIO_API_KEY:
            genai.configure(api_key=settings.GOOGLE_AI_STUDIO_API_KEY)
            self.model = genai.GenerativeModel(settings.LLM_MODEL)
            log.info("✅ Gemini client initialised")
        else:
            self.model = None
            log.warning("⚠️  GOOGLE_AI_STUDIO_API_KEY not set – falling back to dummy output")

    def deconstruct(self, prompt_txt: str, score: float) -> Dict:
        """
        Use Gemini to produce a JSON object matching PromptComponents.
        When the client is not available, return placeholders.
        """
        if not self.model:
            # offline / no-key mode
            return {
                "subject": "Unknown",
                "action_setting": "Unknown",
                "emotional_tone": "Unknown",
                "lighting": "Unknown",
                "color_palette": "Unknown",
                "camera_angle": "Unknown",
                "style_aesthetic": "Unknown",
                "composition_framing": "Unknown",
                "original_prompt": prompt_txt,
                "engagement_score": score,
            }

        system_msg = (
            "You are an expert prompt analyst. "
            "Return a JSON **exactly** matching the given schema."
        )
        user_msg = f"""
Prompt: {prompt_txt}

ENGAGEMENT_SCORE: {score}

Break down the prompt into:
subject, action_setting, emotional_tone, lighting, color_palette,
camera_angle, style_aesthetic, composition_framing
"""
        schema = PromptComponents.model_json_schema()  # OpenAPI-style schema

        cfg = types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=400,
            top_p=0.7,
            top_k=40,
            response_mime_type="application/json",
            response_schema=schema,
        )

        resp = self.model.generate_content(
            contents=[{"role": "system", "parts": [system_msg]},
                      {"role": "user",   "parts": [user_msg]}],
            generation_config=cfg,
        )

        try:
            return json.loads(resp.text)
        except Exception as exc:            # malformed, fallback
            log.error("Gemini JSON parse error: %s", exc)
            return {
                "subject": "Unknown",
                "action_setting": "Unknown",
                "emotional_tone": "Unknown",
                "lighting": "Unknown",
                "color_palette": "Unknown",
                "camera_angle": "Unknown",
                "style_aesthetic": "Unknown",
                "composition_framing": "Unknown",
                "original_prompt": prompt_txt,
                "engagement_score": score,
            }


# ───────────────────────────────
# Main workflow class
# ───────────────────────────────
class PromptDeconstructor:
    def __init__(self):
        self.trend_manager = TrendManager()
        self.llm = GeminiClient()

    # ------------ public ------------
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
            log.warning("No prompts found for the given window.")
            return []

        log.info("🧠 Deconstructing %d prompts with Gemini", len(rows))
        library: List[Dict] = []
        for idx, row in enumerate(rows, 1):
            log.info("[%d/%d] score=%.1f  %.60s", idx, len(rows),
                     row["engagement_score"], row["prompt"].replace("\n", " ")[:60])
            comp = self.llm.deconstruct(row["prompt"], row["engagement_score"])
            library.append(comp)
        return library

    # ------------ helpers ------------
    @staticmethod
    def save_to_file(data: List[Dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        log.info("💾 Saved component library → %s", path)

    @staticmethod
    def analyse(data: List[Dict]) -> None:
        if not data:
            return
        log.info("📊 ANALYSIS\n" + "═" * 50)
        subjects = [d["subject"] for d in data]
        log.info("🎯 Top subjects: %s", ", ".join(list(dict.fromkeys(subjects))[:5]))
        styles = " ".join(d["style_aesthetic"].lower() for d in data)
        common = Counter(styles.split()).most_common(5)
        log.info("🎨 Style keywords: %s",
                 ", ".join(f"{w}({c})" for w, c in common))
        avg_score = sum(d["engagement_score"] for d in data) / len(data)
        log.info("📈 Average engagement score: %.1f", avg_score)


# ───────────────────────────────
# CLI runner
# ───────────────────────────────
def main():
    log.info("🚀 Prompt Component Library Builder")
    decon = PromptDeconstructor()
    library = decon.build_library(hours_back=24, top_n=100,
                                  p_threshold=0.7, num_samples=5)

    if library:
        out_path = SAMPLES_DIR / f"component_library_{datetime.now():%Y%m%d_%H%M}.json"
        decon.save_to_file(library, out_path)
        decon.analyse(library)
        # show the first example
        print("\n📋 EXAMPLE COMPONENT")
        print(json.dumps(library[0], indent=2, ensure_ascii=False))
    else:
        log.warning("No components extracted.")


if __name__ == "__main__":
    main()
