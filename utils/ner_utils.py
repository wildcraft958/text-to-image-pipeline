import spacy
from typing import List, Dict, Set
import re
import subprocess
import sys

class NERProcessor:
    def __init__(self):
        """Initialize NLP processor with spaCy"""
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model with automatic download if needed"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("📦 Installing spaCy English model...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                return spacy.load("en_core_web_sm")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install spaCy model: {e}")
                # Return a minimal processor that doesn't use spaCy
                return None
    
    def extract_entities(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Extract named entities from text and keywords"""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities
            "PRODUCT": [],
            "EVENT": [],
            "ART": [],  # Artworks, books, songs
            "OTHER": []
        }
        
        if not self.nlp:
            return entities
            
        # Combine text and keywords
        full_text = f"{text} {' '.join(keywords)}"
        
        # Process with spaCy
        doc = self.nlp(full_text)
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
            else:
                entities["OTHER"].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def enhance_keywords(self, keywords: List[str]) -> List[str]:
        """Enhance keywords by extracting key phrases and lemmatizing"""
        enhanced = set(keywords)
        
        if not self.nlp:
            return list(enhanced)
        
        for keyword in keywords:
            doc = self.nlp(keyword)
            
            # Add lemmatized versions
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    enhanced.add(token.lemma_.lower())
            
            # Add noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.strip()) > 2:
                    enhanced.add(chunk.text.strip().lower())
        
        return list(enhanced)
    
    def create_cache_key(self, title: str, keywords: List[str], entities: Dict[str, List[str]]) -> str:
        """Create a cache key based on semantic content"""
        # Normalize and sort for consistent keys
        normalized_title = re.sub(r'\W+', ' ', title.lower()).strip()
        normalized_keywords = sorted([re.sub(r'\W+', ' ', kw.lower()).strip() for kw in keywords])
        
        # Include important entities
        important_entities = []
        for entity_type in ["PERSON", "ORG", "GPE", "PRODUCT"]:
            important_entities.extend(entities.get(entity_type, []))
        
        important_entities = sorted([re.sub(r'\W+', ' ', ent.lower()).strip() for ent in important_entities])
        
        # Create hash-like key
        key_components = [normalized_title] + normalized_keywords + important_entities
        cache_key = "_".join(key_components[:10])  # Limit length
        
        return re.sub(r'[^\w_]', '', cache_key)[:50]  # Clean and limit length
