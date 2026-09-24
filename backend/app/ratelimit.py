"""Basit, bellek içi istek sınırlayıcı (tek süreçli backend için).

/ask her çağrıda ücretli LLM API'sini kullanır. Bu sınırlayıcı iki şeyi korur:
- istemci başına (IP) dakika ve gün sınırı: tek bir kullanıcının sistemi meşgul etmesini önler;
- tüm istemciler için günlük toplam sınır: en kötü durumda günlük maliyeti sınırlar.

Sayaçlar bellekte tutulur; sunucu yeniden başlarsa sıfırlanır. Birden fazla süreç/instance
çalıştırılırsa her biri kendi sayacını tutar.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_client: list[tuple[int, int]], global_limit: tuple[int, int] | None = None):
        """per_client: [(izin verilen istek sayısı, pencere saniyesi), ...]; 0 sayılı sınırlar yok sayılır."""
        self.per_client = [(n, w) for n, w in per_client if n > 0]
        self.global_limit = global_limit if global_limit and global_limit[0] > 0 else None
        self._hits: dict[str, deque] = defaultdict(deque)
        self._global: deque = deque()
        self._lock = threading.Lock()
        self._longest = max([w for _, w in self.per_client], default=0)

    @staticmethod
    def _retry_after(hits: deque, limit: int, window: int, now: float) -> int | None:
        recent = sum(1 for t in hits if t > now - window)
        if recent < limit:
            return None
        # En eski penceredeki isteğin süresi dolunca yeniden izin verilir.
        oldest_in_window = [t for t in hits if t > now - window][-limit]
        return max(1, int(oldest_in_window + window - now) + 1)

    def check(self, client: str, now: float | None = None) -> int | None:
        """İzin varsa isteği kaydeder ve None döner; yoksa kaç saniye sonra denenebileceğini döner."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] <= now - self._longest:
                hits.popleft()
            waits = [self._retry_after(hits, n, w, now) for n, w in self.per_client]
            if self.global_limit:
                n, w = self.global_limit
                while self._global and self._global[0] <= now - w:
                    self._global.popleft()
                waits.append(self._retry_after(self._global, n, w, now))
            waits = [w for w in waits if w is not None]
            if waits:
                return max(waits)
            hits.append(now)
            if self.global_limit:
                self._global.append(now)
            if len(self._hits) > 10_000:  # Uzun süre çalışan sunucuda eski istemcileri temizle.
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return None


def client_key(headers, fallback: str | None) -> str:
    """İstemci IP'si. Render, Hugging Face ve Cloud Run bir proxy arkasında çalışır;
    gerçek IP X-Forwarded-For'un ilk değeridir. Bu başlık istemci tarafından taklit
    edilebilir, bu yüzden maliyeti asıl sınırlayan şey global günlük sınırdır."""
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or fallback or "unknown"
