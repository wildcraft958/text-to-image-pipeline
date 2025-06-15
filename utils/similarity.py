from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self):
        """Initialize sentence transformer for semantic similarity"""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def find_similar_prompts(self, 
                           query_text: str, 
                           cached_prompts: List[Tuple[str, str]], 
                           threshold: float = 0.85) -> Optional[str]:
        """Find similar cached prompts above threshold"""
        if not cached_prompts:
            return None
        
        query_embedding = self.model.encode([query_text])
        cached_texts = [prompt[0] for prompt in cached_prompts]
        cached_embeddings = self.model.encode(cached_texts)
        
        similarities = cosine_similarity(query_embedding, cached_embeddings)[0]
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            return cached_prompts[best_idx][1]  # Return the cached prompt
        
        return None
