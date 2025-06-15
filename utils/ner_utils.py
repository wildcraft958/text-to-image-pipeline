import spacy
from typing import List, Dict, Set
import re

class NERProcessor:
    def __init__(self):
        """Initialize NLP processor with spaCy"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Extract named entities from text and keywords"""
        # Combine text and keywords
        full_text = f"{text} {' '.join(keywords)}"
        
        # Process with spaCy
        doc = self.nlp(full_text)
        
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities (countries, cities, states)
            "PRODUCT": [],
            "EVENT": [],
            "ART": [],  # Artworks, books, songs, etc.
            "OTHER": []
        }
        
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
