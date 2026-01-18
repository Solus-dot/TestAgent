import numpy as np
import pickle
import sys
import os
import time
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class VectorMemory:
    """
    Simple but powerful vector memory system for the agent.
    Uses sentence embeddings for semantic similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_path: str = "memory.pkl"):
        """
        Initialize the memory system.
        
        Args:
            model_name: HuggingFace model for embeddings (default is fast and lightweight)
            storage_path: Where to save/load memories
        """
        print(f"[MEMORY] Loading embedding model: {model_name}...", file=sys.stderr)

        # Try to load offline first (Fast & No Internet required)
        try:
            self.encoder = SentenceTransformer(model_name, device='cpu', local_files_only=True)
            print("  > Model loaded from local cache (Offline mode)", file=sys.stderr)
        except Exception:
            # If missing, download it (First run only)
            print("  > Model not found locally. Downloading from Hugging Face...", file=sys.stderr)
            self.encoder = SentenceTransformer(model_name, device='cpu')

        self.storage_path = storage_path
        self.memories = []  # List of memory objects
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Create storage directory if it doesn't exist
        storage_dir = os.path.dirname(storage_path)
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
        
        # Try to load existing memories
        self.load()
        
        print(f"[MEMORY] Memory system ready ({len(self.memories)} existing memories)", file=sys.stderr)
    
    def add(self, text: str, metadata: Optional[Dict] = None, memory_type: str = "conversation"):
        """
        Add a new memory.
        
        Args:
            text: The content to remember
            metadata: Optional additional info (user_id, tags, etc.)
            memory_type: Type of memory (conversation, fact, preference, etc.)
        """
        self.load()

        # Generate embedding
        vector = self.encoder.encode(text, convert_to_numpy=True)
        
        memory = {
            "id": len(self.memories),
            "text": text,
            "vector": vector,
            "metadata": metadata or {},
            "type": memory_type,
            "timestamp": time.time(),
            "access_count": 0,
            "last_accessed": time.time()
        }
        
        self.memories.append(memory)
        print(f"[MEMORY] Added memory #{memory['id']}: {text[:60]}...", file=sys.stderr)
        
        # Save immediately to prevent data loss
        self.save()
    
    def search(self, query: str, top_k: int = 5, memory_type: Optional[str] = None, min_score: float = 0.3) -> List[Dict]:
        """
        Search for relevant memories.
        
        Args:
            query: What to search for
            top_k: How many results to return
            memory_type: Filter by memory type (optional)
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of relevant memories with scores
        """
        self.load()

        if not self.memories:
            return []
        
        # Encode query
        query_vector = self.encoder.encode(query, convert_to_numpy=True)
        
        # Filter by type if specified
        candidates = self.memories
        if memory_type:
            candidates = [m for m in self.memories if m["type"] == memory_type]
        
        if not candidates:
            return []
        
        # Calculate similarities
        similarities = []
        for memory in candidates:
            # Cosine similarity
            similarity = self._cosine_similarity(query_vector, memory["vector"])
            
            if similarity >= min_score:
                similarities.append((similarity, memory))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Update access stats for top results
        results = []
        for score, memory in similarities[:top_k]:
            memory["access_count"] += 1
            memory["last_accessed"] = time.time()
            
            results.append({
                "text": memory["text"],
                "score": float(score),
                "type": memory["type"],
                "metadata": memory["metadata"],
                "id": memory["id"],
                "timestamp": memory["timestamp"]
            })
        
        # Save immediately to prevent data loss
        self.save()
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_by_id(self, memory_id: int) -> Optional[Dict]:
        """Retrieve a specific memory by ID"""
        for memory in self.memories:
            if memory["id"] == memory_id:
                return {
                    "text": memory["text"],
                    "type": memory["type"],
                    "metadata": memory["metadata"],
                    "timestamp": memory["timestamp"]
                }
        return None
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID"""
        self.load()
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                deleted = self.memories.pop(i)
                print(f"[MEMORY] Deleted memory #{memory_id}: {deleted['text'][:60]}...", file=sys.stderr)
                self.save()
                return True
        return False
        
    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        if not self.memories:
            return {
                "total_memories": 0,
                "oldest": None,
                "newest": None,
                "most_accessed": None
            }
        
        sorted_by_time = sorted(self.memories, key=lambda x: x["timestamp"])
        sorted_by_access = sorted(self.memories, key=lambda x: x["access_count"], reverse=True)
        
        return {
            "total_memories": len(self.memories),
            "oldest": {
                "text": sorted_by_time[0]["text"][:60],
                "age_days": (time.time() - sorted_by_time[0]["timestamp"]) / 86400
            },
            "newest": {
                "text": sorted_by_time[-1]["text"][:60],
                "age_days": (time.time() - sorted_by_time[-1]["timestamp"]) / 86400
            },
            "most_accessed": {
                "text": sorted_by_access[0]["text"][:60],
                "count": sorted_by_access[0]["access_count"]
            },
            "types": {t: len([m for m in self.memories if m["type"] == t]) 
                     for t in set(m["type"] for m in self.memories)}
        }
    
    def save(self):
        """Save memories to disk with error handling"""
        try:
            # Write to temporary file first
            temp_path = self.storage_path + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(self.memories, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Only replace original if write was successful
            if os.path.exists(self.storage_path):
                os.remove(self.storage_path)
            os.rename(temp_path, self.storage_path)
            
            print(f"[MEMORY] Saved {len(self.memories)} memories to {self.storage_path}", file=sys.stderr)
        except Exception as e:
            print(f"[MEMORY] Failed to save memories: {e}", file=sys.stderr)
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def load(self):
        """Load memories from disk with error handling"""
        if not os.path.exists(self.storage_path):
            print(f"[MEMORY] No existing memories found, starting fresh", file=sys.stderr)
            self.memories = []
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                self.memories = pickle.load(f)
            print(f"[MEMORY] Loaded {len(self.memories)} memories from {self.storage_path}", file=sys.stderr)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"[MEMORY] Corrupted memory file: {e}", file=sys.stderr)
            # Backup corrupted file
            backup_path = self.storage_path + ".corrupted"
            if os.path.exists(self.storage_path):
                os.rename(self.storage_path, backup_path)
                print(f"[MEMORY] Backed up corrupted file to {backup_path}", file=sys.stderr)
            self.memories = []
        except Exception as e:
            print(f"[MEMORY] Failed to load memories: {e}", file=sys.stderr)
            self.memories = []
    
    def clear(self):
        """Clear all memories (use with caution!)"""
        self.memories = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        print("[MEMORY] All memories cleared", file=sys.stderr)
    
    def export_txt(self, filepath: str = "memories/memories_export.txt"):
        """Export memories to a readable text file"""
        # Create directory if it doesn't exist
        export_dir = os.path.dirname(filepath)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== AGENT MEMORIES ===\n\n")
            
            for memory in sorted(self.memories, key=lambda x: x["timestamp"], reverse=True):
                f.write(f"ID: {memory['id']}\n")
                f.write(f"Type: {memory['type']}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(memory['timestamp']))}\n")
                f.write(f"Accessed: {memory['access_count']} times\n")
                f.write(f"Text: {memory['text']}\n")
                if memory['metadata']:
                    f.write(f"Metadata: {memory['metadata']}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"[MEMORY] Exported {len(self.memories)} memories to {filepath}", file=sys.stderr)
