"""
utils/story_director.py
Phase-2 · Coherent Narrative Synthesis
─────────────────────────────────────
• Reads the latest *component_library_*.json from the top-level samples/ folder
  (written by utils/prompt_deconstructor.py).
• Uses the **new** Google Gen AI SDK (≥ 1.0) – no more genai.configure().
• Produces a 3-panel StoryPlan (pydantic) with Chain-of-Thought prompting.
• Writes the result into samples/story_plan_YYYYMMDD_HHMM.json.

Only SDK-related and path tweaks have been made; everything else is
unchanged from the previous logic.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from google import genai                         # ← new SDK import
from google.genai import types

from pydantic import BaseModel, Field
from config.settings import settings                   # unified config

# ───────────────────────────────
# Logging
# ───────────────────────────────
log = logging.getLogger("story-director")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────
# Pydantic output schema
# ───────────────────────────────
class Character(BaseModel):
    name: str
    description: str
    archetype: str


class StoryPanel(BaseModel):
    panel: int
    narrative_beat: str
    location: str
    emotion: str
    character_state: str
    key_objects: List[str]
    lighting: str
    camera_angle: str
    color_palette: str


class StoryPlan(BaseModel):
    title: str
    theme: str
    character: Character
    panels: List[StoryPanel]
    narrative_arc: str


# ───────────────────────────────
# StoryDirector
# ───────────────────────────────
class StoryDirector:
    def __init__(self, component_library_path: str | None = None):
        # Gemini client (new SDK)
        if settings.GOOGLE_AI_STUDIO_API_KEY:
            self.client = genai.Client(api_key=settings.GOOGLE_AI_STUDIO_API_KEY)
            log.info("✅ GenAI client initialised (model %s)", settings.LLM_MODEL)
        else:
            self.client = None
            log.warning("⚠️  GOOGLE_AI_STUDIO_API_KEY not set – running offline")

        # locate component library
        self.component_library_path = (
            component_library_path or self._find_latest_component_lib()
        )
        self.component_library = self._load_component_library()

    # ---------- private ----------
    def _find_latest_component_lib(self) -> str:
        libs = sorted(SAMPLES_DIR.glob("component_library_*.json"))
        if not libs:
            log.warning("⚠️  No component library found in %s", SAMPLES_DIR)
            return ""
        return str(libs[-1])

    def _load_component_library(self) -> List[Dict]:
        if not self.component_library_path:
            return []
        try:
            with open(self.component_library_path, encoding="utf-8") as fh:
                data = json.load(fh)
            log.info("✅ Loaded %d components (%s)",
                     len(data), Path(self.component_library_path).name)
            return data
        except Exception as exc:
            log.error("❌ Cannot read component library: %s", exc)
            return []

    # ---------- element extraction ----------
    @staticmethod
    def _extract_elements(comps: List[Dict]) -> Dict[str, List[str]]:
        buckets = {
            "subjects": [], "actions": [], "emotions": [], "lighting": [],
            "palette": [], "camera": [], "aesthetics": []
        }
        for c in comps:
            buckets["subjects"].append(c.get("subject", ""))
            buckets["actions"].append(c.get("action_setting", ""))
            buckets["emotions"].append(c.get("emotional_tone", ""))
            buckets["lighting"].append(c.get("lighting", ""))
            buckets["palette"].append(c.get("color_palette", ""))
            buckets["camera"].append(c.get("camera_angle", ""))
            buckets["aesthetics"].append(c.get("style_aesthetic", ""))

        return {k: list({v for v in vals if v})[:5] for k, vals in buckets.items()}

    # ---------- prompt builder ----------
    def _cot_prompt(self, e: Dict[str, List[str]]) -> str:
        return f"""[SYSTEM]
You are a master storyteller. Produce a 3-panel visual story concept in JSON
matching the provided schema. First think step-by-step (Chain-of-Thought),
then output ONLY the JSON.

[USER]
Trending Creative Elements
• Subjects: {', '.join(e['subjects'])}
• Actions: {', '.join(e['actions'])}
• Emotions: {', '.join(e['emotions'])}
• Styles: {', '.join(e['aesthetics'])}
• Lighting: {', '.join(e['lighting'])}
• Palette: {', '.join(e['palette'])}

Return a StoryPlan as JSON.
"""

    # ---------- generation ----------
    def generate_story_plan(self, num_components: int = 12) -> Optional[StoryPlan]:
        if not self.component_library:
            log.error("No component library loaded")
            return None

        elements = self._extract_elements(self.component_library[:num_components])
        prompt = self._cot_prompt(elements)

        # offline stub
        if not self.client:
            return StoryPlan(
                title="Stub Story",
                theme="Creativity",
                character=Character(
                    name="Alex", description="Curious explorer", archetype="Seeker"
                ),
                narrative_arc="Stub arc",
                panels=[
                    StoryPanel(
                        panel=1, narrative_beat="Setup", location="Nowhere",
                        emotion="Neutral", character_state="Stable",
                        key_objects=["None"], lighting="Flat",
                        camera_angle="Wide", color_palette="Muted"
                    )
                ],
            )

        try:
            gen_cfg = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1500,
                top_p=0.8,
                top_k=40,
                response_mime_type="application/json",
                response_schema=StoryPlan.model_json_schema(),
            )

            resp = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=gen_cfg,
            )

            data = json.loads(resp.text)
            plan = StoryPlan(**data)
            log.info("✅ Story plan generated: %s", plan.title)
            return plan
        except Exception as exc:
            log.error("❌ GenAI generation error: %s", exc)
            return None

    # ---------- I/O ----------
    @staticmethod
    def save(plan: StoryPlan, path: Path):
        with path.open("w", encoding="utf-8") as fh:
            json.dump(plan.model_dump(), fh, indent=2, ensure_ascii=False)
        log.info("💾 Saved story plan → %s", path)

    @staticmethod
    def display(plan: StoryPlan):
        print(f"\n🎬 STORY: {plan.title}\n{'='*60}")
        print(f"Theme: {plan.theme}")
        print(f"Character: {plan.character.name} – {plan.character.archetype}")
        print(f"Narrative Arc: {plan.narrative_arc}\n")
        for p in plan.panels:
            print(f"-- Panel {p.panel}: {p.narrative_beat}")
            print(f"   Location: {p.location} | Emotion: {p.emotion}")
            print(f"   Objects: {', '.join(p.key_objects)}")
            print(f"   Lighting: {p.lighting} | Camera: {p.camera_angle}")
            print(f"   Palette: {p.color_palette}\n")

    @staticmethod
    def analyse(plan: StoryPlan):
        moods = Counter(panel.emotion.lower() for panel in plan.panels)
        print("Mood distribution:", moods)


# ───────────────────────────────
# CLI runner
# ───────────────────────────────
def main():
    log.info("🚀 Phase 2 – Coherent Narrative Synthesis")
    director = StoryDirector()
    story = director.generate_story_plan()

    if story:
        director.display(story)
        out = SAMPLES_DIR / f"story_plan_{datetime.now():%Y%m%d_%H%M}.json"
        director.save(story, out)
        director.analyse(story)
    else:
        log.error("Story generation failed.")


if __name__ == "__main__":
    main()
