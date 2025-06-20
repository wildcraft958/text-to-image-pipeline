import os
import json
import logging
from typing import List, Dict, Type, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Output ---

class Character(BaseModel):
    """Character definition for the story."""
    name: str = Field(description="The protagonist's name")
    description: str = Field(description="Brief character description highlighting key traits")
    archetype: str = Field(description="Character archetype (e.g., 'reluctant hero', 'wise mentor')")

class StoryPanel(BaseModel):
    """Individual panel/scene in the story."""
    panel: int = Field(description="Panel number (1, 2, or 3)")
    narrative_beat: str = Field(description="The story beat (Setup, Confrontation, Resolution)")
    location: str = Field(description="Where this scene takes place")
    emotion: str = Field(description="Dominant emotional tone of the scene")
    character_state: str = Field(description="Character's physical and emotional state")
    key_objects: List[str] = Field(description="Important visual elements in the scene")
    lighting: str = Field(description="Lighting conditions and mood")
    camera_angle: str = Field(description="Suggested camera perspective")
    color_palette: str = Field(description="Dominant colors for this panel")

class StoryPlan(BaseModel):
    """Complete story plan structure."""
    title: str = Field(description="Story title")
    theme: str = Field(description="Central theme or message")
    character: Character = Field(description="Main character details")
    panels: List[StoryPanel] = Field(description="Three-panel story structure")
    narrative_arc: str = Field(description="Brief summary of the character's journey")

# --- Story Director Class ---

class StoryDirector:
    """
    Phase 2: Coherent Narrative Synthesis
    Creates coherent story plans from deconstructed prompt components using Chain of Thought reasoning.
    """
    
    def __init__(self, component_library_path: str = "component_library.json"):
        """
        Initialize the Story Director with Gemini AI client.
        
        Args:
            component_library_path: Path to the component library JSON file from Phase 1
        """
        try:
            # Initialize Gemini client
            self.gemini_client = genai.Client(api_key=settings.GOOGLE_AI_STUDIO_API_KEY)
            self.model_name = settings.LLM_MODEL
            self.temperature = 0.7  # Higher for creative storytelling
            
            self.generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=1500,
                top_p=0.8,
                top_k=40
            )
            
            logger.info(f"✅ Story Director initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise
            
        # Load component library
        self.component_library = self._load_component_library(component_library_path)
    
    def _load_component_library(self, path: str) -> List[Dict]:
        """Load the component library from Phase 1."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                components = json.load(f)
            logger.info(f"✅ Loaded {len(components)} components from library")
            return components
        except FileNotFoundError:
            logger.warning(f"⚠️ Component library not found at {path}. Using empty library.")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading component library: {e}")
            return []
    
    def _extract_story_elements(self, components: List[Dict]) -> Dict:
        """Extract and categorize story elements from components."""
        subjects = []
        actions = []
        settings = []
        emotions = []
        lighting_styles = []
        color_palettes = []
        camera_angles = []
        aesthetics = []
        
        for comp in components:
            if comp.get('subject'):
                subjects.append(comp['subject'])
            if comp.get('action_setting'):
                actions.append(comp['action_setting'])
            if comp.get('emotional_tone'):
                emotions.append(comp['emotional_tone'])
            if comp.get('lighting'):
                lighting_styles.append(comp['lighting'])
            if comp.get('color_palette'):
                color_palettes.append(comp['color_palette'])
            if comp.get('camera_angle'):
                camera_angles.append(comp['camera_angle'])
            if comp.get('style_aesthetic'):
                aesthetics.append(comp['style_aesthetic'])
        
        return {
            'subjects': list(set(subjects))[:5],
            'actions': list(set(actions))[:5],
            'emotions': list(set(emotions))[:5],
            'lighting_styles': list(set(lighting_styles))[:3],
            'color_palettes': list(set(color_palettes))[:3],
            'camera_angles': list(set(camera_angles))[:3],
            'aesthetics': list(set(aesthetics))[:3]
        }
    
    def create_cot_prompt(self, story_elements: Dict) -> str:
        """
        Create a Chain of Thought prompt for story generation.
        This follows the CoT pattern from the tutorial attachment.
        """
        return f"""[SYSTEM]
You are a master storyteller and narrative architect. Your task is to create a coherent 3-panel visual story concept based on trending creative elements. You must use Chain of Thought reasoning to establish a single protagonist, consistent setting, and a clear 3-act plot structure with causal connections between each panel.

First, think step-by-step through your creative process, then output a structured plan in JSON format.

[USER]
Trending Creative Elements from High-Engagement Content:

**Subjects/Characters:** {', '.join(story_elements.get('subjects', []))}
**Actions/Scenarios:** {', '.join(story_elements.get('actions', []))}
**Emotional Tones:** {', '.join(story_elements.get('emotions', []))}
**Visual Styles:** {', '.join(story_elements.get('aesthetics', []))}
**Lighting Approaches:** {', '.join(story_elements.get('lighting_styles', []))}
**Color Palettes:** {', '.join(story_elements.get('color_palettes', []))}

Please generate a coherent 3-panel story plan that creates a meaningful narrative arc.

[ASSISTANT]
Chain of Thought:

1. **Character Conception:** I need to create a single protagonist who can naturally experience the emotional journey suggested by these elements. Looking at the subjects and emotional tones, I'll develop a character who can embody the core themes while maintaining consistency across all three panels.

2. **Thematic Analysis:** I'll identify the central theme that connects these elements. What universal human experience or journey do these trending elements suggest? This theme will be the backbone of my story.

3. **3-Act Structure Planning:**
   - **Act 1 (Setup - Panel 1):** Establish the character's initial state and the inciting incident. This sets up the emotional stakes and motivation.
   - **Act 2 (Confrontation - Panel 2):** The main conflict or challenge. This should directly result from Panel 1's setup and create tension that demands resolution.
   - **Act 3 (Resolution - Panel 3):** The outcome of the confrontation. This should provide emotional catharsis and show character growth or change.

4. **Visual Continuity:** I'll ensure each panel uses complementary visual elements (lighting, colors, composition) that support the narrative progression while maintaining aesthetic coherence.

5. **Causal Connections:** Each panel must logically lead to the next, creating a story where Panel 2 happens *because* of Panel 1, and Panel 3 happens *because* of Panel 2.

Now I'll structure this reasoning into the requested JSON format:"""

    def generate_story_plan(self, num_components: int = 10) -> Optional[StoryPlan]:
        """
        Generate a coherent story plan using Chain of Thought reasoning.
        
        Args:
            num_components: Number of components to use from the library
            
        Returns:
            StoryPlan object or None if generation fails
        """
        if not self.component_library:
            logger.error("❌ No component library available")
            return None
        
        # Select diverse components
        selected_components = self.component_library[:num_components]
        story_elements = self._extract_story_elements(selected_components)
        
        # Create CoT prompt
        cot_prompt = self.create_cot_prompt(story_elements)
        
        try:
            # Configure for structured JSON output
            generation_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1500,
                top_p=0.8,
                top_k=40,
                response_mime_type="application/json",
                response_schema=StoryPlan,
            )
            
            # Generate story plan
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=cot_prompt,
                config=generation_config
            )
            
            # Parse response
            story_data = json.loads(response.text)
            story_plan = StoryPlan(**story_data)
            
            logger.info(f"✅ Generated story plan: '{story_plan.title}'")
            return story_plan
            
        except Exception as e:
            logger.error(f"❌ Error generating story plan: {e}")
            return None
    
    def save_story_plan(self, story_plan: StoryPlan, filename: str = "story_plan.json"):
        """Save the generated story plan to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(story_plan.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Story plan saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving story plan: {e}")
    
    def display_story_plan(self, story_plan: StoryPlan):
        """Display the story plan in a formatted way."""
        print(f"\n🎬 STORY PLAN: '{story_plan.title}'")
        print("=" * 60)
        print(f"📖 Theme: {story_plan.theme}")
        print(f"🎭 Character: {story_plan.character.name} - {story_plan.character.description}")
        print(f"📈 Narrative Arc: {story_plan.narrative_arc}")
        
        print(f"\n📋 PANELS:")
        print("-" * 40)
        
        for panel in story_plan.panels:
            print(f"\n🎬 PANEL {panel.panel}: {panel.narrative_beat}")
            print(f"   📍 Location: {panel.location}")
            print(f"   😊 Emotion: {panel.emotion}")
            print(f"   🎭 Character State: {panel.character_state}")
            print(f"   🎨 Visual Elements: {', '.join(panel.key_objects)}")
            print(f"   💡 Lighting: {panel.lighting}")
            print(f"   📷 Camera: {panel.camera_angle}")
            print(f"   🎨 Colors: {panel.color_palette}")

def main():
    """
    Main function to run the Story Director workflow.
    """
    print("🎬 Starting Phase 2: Coherent Narrative Synthesis")
    print("=" * 60)
    
    try:
        # Initialize Story Director
        story_director = StoryDirector("component_library.json")
        
        # Generate story plan
        print("🧠 Generating coherent story plan using Chain of Thought...")
        story_plan = story_director.generate_story_plan(num_components=15)
        
        if story_plan:
            # Display the plan
            story_director.display_story_plan(story_plan)
            
            # Save the plan
            story_director.save_story_plan(story_plan, "coherent_story_plan.json")
            
            print(f"\n🎉 Successfully created coherent narrative: '{story_plan.title}'")
            print(f"📊 Story uses elements from trending content to ensure engagement")
            print(f"🔗 All panels are causally connected for narrative coherence")
            
        else:
            print("❌ Failed to generate story plan")
            
    except Exception as e:
        logger.error(f"❌ Error in main workflow: {e}")
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
# Run the main function if this script is executed directly