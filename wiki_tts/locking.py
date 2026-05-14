import redis
from wiki_tts.config import REDIS_URL, LOCK_TTL

redis_client = redis.from_url(REDIS_URL)

_LOCK_PREFIX = "wiki-tts:lock:"


def _lock_key(safe_article: str, sec_title: str) -> str:
    return f"{_LOCK_PREFIX}{safe_article}:{sec_title}"


def acquire_lock(safe_article: str, sec_title: str) -> bool:
    """
    Try to acquire a Redis lock for a section.
    Returns True if this call acquired the lock (caller should queue the task).
    Returns False if the lock is already held (task already queued by another process).
    """
    key = _lock_key(safe_article, sec_title)
    if redis_client.setnx(key, "1"):
        redis_client.expire(key, LOCK_TTL)
        return True
    return False


def release_lock(safe_article: str, sec_title: str) -> None:
    """Release a Redis lock."""
    redis_client.delete(_lock_key(safe_article, sec_title))
