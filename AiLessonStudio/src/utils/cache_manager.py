import pickle
import hashlib
import time
from typing import Any, Optional
from pathlib import Path


class CacheManager:
    """Simple file-based cache system"""

    def __init__(self, cache_dir: Path, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl  # Time to live in seconds
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_file = self._get_cache_path(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Check if cache expired
            if time.time() - data['timestamp'] > self.ttl:
                self.delete(key)
                return None

            return data['value']

        except:
            return None

    def set(self, key: str, value: Any):
        """Set cache value"""
        cache_file = self._get_cache_path(key)

        data = {
            'timestamp': time.time(),
            'value': value,
            'key': key
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except:
            return False

    def delete(self, key: str):
        """Delete cache entry"""
        cache_file = self._get_cache_path(key)
        try:
            cache_file.unlink()
            return True
        except:
            return False

    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        files = list(self.cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in files)

        return {
            'total_entries': len(files),
            'total_size_kb': total_size / 1024,
            'cache_dir': str(self.cache_dir)
        }