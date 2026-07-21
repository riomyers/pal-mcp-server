"""
Storage backends for conversation threads

This module provides storage backends for persisting conversation contexts:
- InMemoryStorage: Fast, process-local storage (default, ephemeral)
- SQLiteStorage: File-backed persistent storage that survives server restarts

Configure via STORAGE_BACKEND env var:
  STORAGE_BACKEND=memory  (default)
  STORAGE_BACKEND=sqlite

Key Features:
- Thread-safe operations
- TTL support with automatic expiration
- Background cleanup for memory management
- Singleton pattern for consistent state
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from utils.env import get_env

logger = logging.getLogger(__name__)


class InMemoryStorage:
    """Thread-safe in-memory storage for conversation threads"""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        # Match Redis behavior: cleanup interval based on conversation timeout
        # Run cleanup at 1/10th of timeout interval (e.g., 18 mins for 3 hour timeout)
        timeout_hours = int(get_env("CONVERSATION_TIMEOUT_HOURS", "3") or "3")
        self._cleanup_interval = (timeout_hours * 3600) // 10
        self._cleanup_interval = max(300, self._cleanup_interval)  # Minimum 5 minutes
        self._shutdown = False

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        logger.info(
            f"In-memory storage initialized with {timeout_hours}h timeout, cleanup every {self._cleanup_interval//60}m"
        )

    def set_with_ttl(self, key: str, ttl_seconds: int, value: str) -> None:
        """Store value with expiration time"""
        with self._lock:
            expires_at = time.time() + ttl_seconds
            self._store[key] = (value, expires_at)
            logger.debug(f"Stored key {key} with TTL {ttl_seconds}s")

    def get(self, key: str) -> Optional[str]:
        """Retrieve value if not expired"""
        with self._lock:
            if key in self._store:
                value, expires_at = self._store[key]
                if time.time() < expires_at:
                    logger.debug(f"Retrieved key {key}")
                    return value
                else:
                    # Clean up expired entry
                    del self._store[key]
                    logger.debug(f"Key {key} expired and removed")
        return None

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        """Redis-compatible setex method"""
        self.set_with_ttl(key, ttl_seconds, value)

    def _cleanup_worker(self):
        """Background thread that periodically cleans up expired entries"""
        while not self._shutdown:
            time.sleep(self._cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove all expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [k for k, (_, exp) in self._store.items() if exp < current_time]
            for key in expired_keys:
                del self._store[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired conversation threads")

    def shutdown(self):
        """Graceful shutdown of background thread"""
        self._shutdown = True
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)


class SQLiteStorage:
    """SQLite-backed persistent storage for conversation threads.

    Survives server restarts and can be shared across processes.
    Stores data in ~/.pal/conversations.db by default.
    """

    def __init__(self, db_path: str | None = None):
        import sqlite3  # Lazy import — only needed for SQLite backend

        self._sqlite3 = sqlite3

        if db_path is None:
            pal_dir = Path.home() / ".pal"
            pal_dir.mkdir(exist_ok=True)
            db_path = str(pal_dir / "conversations.db")

        self._db_path = db_path
        self._local = threading.local()
        self._shutdown = False

        # Initialize the table
        conn = self._get_conn()
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL NOT NULL
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON conversations(expires_at)")
        conn.commit()

        # Background cleanup
        timeout_hours = int(get_env("CONVERSATION_TIMEOUT_HOURS", "3") or "3")
        self._cleanup_interval = max(300, (timeout_hours * 3600) // 10)

        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"SQLite storage initialized at {db_path}, cleanup every {self._cleanup_interval // 60}m")

    def _get_conn(self):
        """Get a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._sqlite3.connect(self._db_path, timeout=10)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def set_with_ttl(self, key: str, ttl_seconds: int, value: str) -> None:
        """Store value with expiration time"""
        conn = self._get_conn()
        expires_at = time.time() + ttl_seconds
        conn.execute(
            "INSERT OR REPLACE INTO conversations (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, expires_at),
        )
        conn.commit()
        logger.debug(f"Stored key {key} with TTL {ttl_seconds}s (SQLite)")

    def get(self, key: str) -> Optional[str]:
        """Retrieve value if not expired"""
        conn = self._get_conn()
        row = conn.execute("SELECT value, expires_at FROM conversations WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if time.time() < expires_at:
            logger.debug(f"Retrieved key {key} (SQLite)")
            return value
        # Expired — clean it up
        conn.execute("DELETE FROM conversations WHERE key = ?", (key,))
        conn.commit()
        logger.debug(f"Key {key} expired and removed (SQLite)")
        return None

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        """Redis-compatible setex method"""
        self.set_with_ttl(key, ttl_seconds, value)

    def _cleanup_worker(self):
        """Background thread that periodically cleans up expired entries"""
        while not self._shutdown:
            time.sleep(self._cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove all expired entries"""
        try:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM conversations WHERE expires_at < ?", (time.time(),))
            if cursor.rowcount > 0:
                conn.commit()
                logger.debug(f"Cleaned up {cursor.rowcount} expired conversation threads (SQLite)")
        except Exception as exc:
            logger.debug(f"SQLite cleanup error: {exc}")

    def shutdown(self):
        """Graceful shutdown"""
        self._shutdown = True
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)
        try:
            if hasattr(self._local, "conn") and self._local.conn:
                self._local.conn.close()
        except Exception:
            pass


# Global singleton instance
_storage_instance = None
_storage_lock = threading.Lock()


def get_storage_backend() -> InMemoryStorage | SQLiteStorage:
    """Get the global storage instance (singleton pattern).

    Set STORAGE_BACKEND=sqlite to use persistent SQLite storage.
    Default is in-memory (ephemeral, process-local).
    """
    global _storage_instance
    if _storage_instance is None:
        with _storage_lock:
            if _storage_instance is None:
                backend = get_env("STORAGE_BACKEND", "memory") or "memory"
                if backend.lower() == "sqlite":
                    _storage_instance = SQLiteStorage()
                    logger.info("Initialized SQLite conversation storage")
                else:
                    _storage_instance = InMemoryStorage()
                    logger.info("Initialized in-memory conversation storage")
    return _storage_instance
