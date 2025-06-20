"""
utils/story_generator.py
Phase-3 · High-Quality Image Generation
────────────────────────────────────────
• Reads the latest *story_plan_*.json from the top-level samples/ folder.
• Implements the "Character Priming" technique for character consistency.
• For each panel, it assembles a detailed prompt combining the character
  description with the panel's specific narrative and visual elements.
• Uses Google's Vertex AI Imagen model to generate a high-quality image.
• Saves the resulting images into a timestamped folder in outputs/.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Import shared Pydantic models and settings from the project
from config.settings import settings
from utils.story_director import StoryPlan, StoryPanel

# ───────────────────────────────
# Logging
# ───────────────────────────────
logger = logging.getLogger("image-generator")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# ───────────────────────────────
# Path Setup
# ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class StoryGenerator:
    """
    Generates a sequence of images based on a story plan using Vertex AI.
    """

    def __init__(self, story_plan_path: Optional[Path] = None):
        """
        Initializes the generator, loads the story plan, and sets up Vertex AI.
        """
        self.story_plan_path = story_plan_path or self._find_latest_story_plan()
        if not self.story_plan_path:
            logger.error("❌ No story plan found. Aborting.")
            self.story_plan = None
            self.image_model = None
            return

        self.story_plan = self._load_story_plan(self.story_plan_path)
        self.output_dir = self._create_output_directory()
        self._initialize_vertexai()

    def _find_latest_story_plan(self) -> Optional[Path]:
        """Finds the most recently created story_plan JSON file."""
        try:
            latest_plan = max(SAMPLES_DIR.glob("story_plan_*.json"), key=os.path.getctime)
            logger.info(f"✅ Found latest story plan: {latest_plan.name}")
            return latest_plan
        except ValueError:
            return None

    def _load_story_plan(self, plan_path: Path) -> Optional[StoryPlan]:
        """Loads and validates the story plan from a JSON file."""
        try:
            with plan_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            story_plan = StoryPlan(**data)
            logger.info(f"✅ Successfully loaded and parsed story: '{story_plan.title}'")
            return story_plan
        except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
            logger.error(f"❌ Failed to load or parse story plan '{plan_path}': {e}")
            return None

    def _create_output_directory(self) -> Path:
        """Creates a unique, timestamped directory for the output images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize title for use in a directory name
        safe_title = "".join(c for c in self.story_plan.title if c.isalnum() or c in " -_").rstrip()
        run_dir = OUTPUT_DIR / f"{timestamp}_{safe_title}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🎨 Output directory created: {run_dir}")
        return run_dir

    def _initialize_vertexai(self):
        """Initializes the Vertex AI client and the Imagen model."""
        try:
            # The SDK will automatically use the credentials from the
            # GOOGLE_APPLICATION_CREDENTIALS environment variable.
            vertexai.init(project=settings.GOOGLE_PROJECT_ID, location=settings.GOOGLE_LOCATION)
            logger.info("✅ Vertex AI initialized successfully.")

            self.image_model = ImageGenerationModel.from_pretrained(settings.IMAGE_MODEL)
            logger.info(f"✅ Image Model loaded: {settings.IMAGE_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI or load Image model: {e}")
            self.image_model = None

    def _assemble_prompt(self, panel: StoryPanel) -> str:
        """
        Assembles a detailed prompt for a single panel using character priming.
        """
        character_desc = self.story_plan.character.description

        # Combine all panel elements into a coherent scene description
        scene_details = (
            f"The scene takes place at {panel.location}. "
            f"The character is currently {panel.character_state}. "
            f"The overall emotion is {panel.emotion}. "
            f"Key objects in the scene include: {', '.join(panel.key_objects)}. "
        )

        style_details = (
            f"The lighting is {panel.lighting}. "
            f"The camera angle is a {panel.camera_angle}. "
            f"The color palette consists of {panel.color_palette}. "
        )

        # Final prompt construction based on best practices
        prompt = (
            f"cinematic film still of ({character_desc}:1.2), {self.story_plan.character.name}. "
            f"{scene_details} "
            f"{style_details} "
            f"Style: photorealistic, highly detailed, dramatic composition."
        )

        logger.info(f"📜 Assembled prompt for Panel {panel.panel}: {prompt}")
        return prompt

    def generate_images(self):
        """
        Generates images for each panel in the story plan.
        """
        if not self.story_plan or not self.image_model:
            logger.error("❌ Cannot generate images. Story plan or model not available.")
            return

        logger.info(f"🎬 Starting image generation for '{self.story_plan.title}'...")

        for panel in self.story_plan.panels:
            logger.info(f"--- Generating Panel {panel.panel}/{len(self.story_plan.panels)} ---")
            prompt = self._assemble_prompt(panel)

            try:
                # Generate the image using the Vertex AI Imagen model
                images = self.image_model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio="16:9",  # Cinematic aspect ratio
                    safety_filter_level="block_some",
                    person_generation="allow_adult"
                )

                image = images[0]
                output_path = self.output_dir / f"panel_{panel.panel}.png"
                
                # Save the generated image to the output directory
                image.save(location=str(output_path), include_generation_parameters=True)
                logger.info(f"✅ Successfully saved image to {output_path}")

            except Exception as e:
                logger.error(f"❌ Failed to generate or save image for Panel {panel.panel}: {e}")

        logger.info("✨ All panels have been generated.")


def main():
    """CLI entry point to run the image generation process."""
    logger.info("🚀 Phase 3 – High-Quality Image Generation")
    generator = StoryGenerator()
    if generator.story_plan:
        generator.generate_images()
    else:
        logger.error("Image generation could not start due to setup errors.")


if __name__ == "__main__":
    main()
