import sqlite3
import json
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
import hashlib

class CacheService:
    def __init__(self, db_path: str = "prompt_cache.db"):
        """Initialize SQLite cache"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create prompts cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT UNIQUE,
            original_input TEXT,
            generated_prompt TEXT,
            entities TEXT,
            created_at TIMESTAMP,
            last_used TIMESTAMP,
            use_count INTEGER DEFAULT 1
        )
        ''')
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_key ON prompt_cache(cache_key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_used ON prompt_cache(last_used)')
        
        conn.commit()
        conn.close()
    
    def get_cached_prompt(self, cache_key: str) -> Optional[str]:
        """Get cached prompt by key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT generated_prompt FROM prompt_cache 
        WHERE cache_key = ? AND last_used > ?
        ''', (cache_key, datetime.now() - timedelta(hours=24)))
        
        result = cursor.fetchone()
        
        if result:
            # Update usage statistics
            cursor.execute('''
            UPDATE prompt_cache 
            SET last_used = ?, use_count = use_count + 1 
            WHERE cache_key = ?
            ''', (datetime.now(), cache_key))
            conn.commit()
        
        conn.close()
        return result[0] if result else None
    
    def cache_prompt(self, 
                    cache_key: str, 
                    original_input: str, 
                    generated_prompt: str, 
                    entities: Dict[str, List[str]]):
        """Cache a generated prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO prompt_cache 
            (cache_key, original_input, generated_prompt, entities, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cache_key, 
                original_input, 
                generated_prompt,
                json.dumps(entities),
                datetime.now(),
                datetime.now()
            ))
            conn.commit()
        except Exception as e:
            print(f"Error caching prompt: {e}")
        finally:
            conn.close()
    
    def get_similar_prompts(self, limit: int = 100) -> List[Tuple[str, str]]:
        """Get recent prompts for similarity matching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT original_input, generated_prompt 
        FROM prompt_cache 
        WHERE last_used > ?
        ORDER BY use_count DESC, last_used DESC 
        LIMIT ?
        ''', (datetime.now() - timedelta(days=7), limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def cleanup_old_cache(self, days: int = 30):
        """Clean up old cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        DELETE FROM prompt_cache 
        WHERE last_used < ?
        ''', (datetime.now() - timedelta(days=days),))
        
        conn.commit()
        conn.close()
