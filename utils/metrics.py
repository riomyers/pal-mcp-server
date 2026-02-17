"""
Lightweight metrics collection for PAL MCP Server.

Tracks per-tool latency, call counts, error rates, and token usage.
All data is in-memory (resets on server restart). Thread-safe.
"""

import threading
import time
from typing import Any


class Metrics:
    """Thread-safe metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tool_calls: dict[str, int] = {}
        self._tool_errors: dict[str, int] = {}
        self._tool_latency_sum: dict[str, float] = {}
        self._tool_latency_max: dict[str, float] = {}
        self._tool_tokens_in: dict[str, int] = {}
        self._tool_tokens_out: dict[str, int] = {}
        self._start_time = time.time()

    def record_call(self, tool_name: str, duration_ms: float, error: bool = False) -> None:
        """Record a tool call with its duration."""
        with self._lock:
            self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1
            if error:
                self._tool_errors[tool_name] = self._tool_errors.get(tool_name, 0) + 1
            self._tool_latency_sum[tool_name] = self._tool_latency_sum.get(tool_name, 0.0) + duration_ms
            current_max = self._tool_latency_max.get(tool_name, 0.0)
            if duration_ms > current_max:
                self._tool_latency_max[tool_name] = duration_ms

    def record_tokens(self, tool_name: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record token usage for a tool call."""
        with self._lock:
            self._tool_tokens_in[tool_name] = self._tool_tokens_in.get(tool_name, 0) + input_tokens
            self._tool_tokens_out[tool_name] = self._tool_tokens_out.get(tool_name, 0) + output_tokens

    def get_summary(self) -> dict[str, Any]:
        """Get a snapshot of all metrics."""
        with self._lock:
            uptime_seconds = time.time() - self._start_time
            total_calls = sum(self._tool_calls.values())
            total_errors = sum(self._tool_errors.values())

            tools = {}
            for name in sorted(self._tool_calls.keys()):
                calls = self._tool_calls[name]
                errors = self._tool_errors.get(name, 0)
                avg_ms = self._tool_latency_sum.get(name, 0) / calls if calls > 0 else 0
                max_ms = self._tool_latency_max.get(name, 0)
                tokens_in = self._tool_tokens_in.get(name, 0)
                tokens_out = self._tool_tokens_out.get(name, 0)

                tools[name] = {
                    "calls": calls,
                    "errors": errors,
                    "error_rate": f"{(errors / calls * 100):.1f}%" if calls > 0 else "0.0%",
                    "avg_latency_ms": round(avg_ms),
                    "max_latency_ms": round(max_ms),
                    "total_tokens_in": tokens_in,
                    "total_tokens_out": tokens_out,
                }

            return {
                "uptime_seconds": round(uptime_seconds),
                "uptime_human": _format_duration(uptime_seconds),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "tools": tools,
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._tool_calls.clear()
            self._tool_errors.clear()
            self._tool_latency_sum.clear()
            self._tool_latency_max.clear()
            self._tool_tokens_in.clear()
            self._tool_tokens_out.clear()
            self._start_time = time.time()


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


# Global singleton
_metrics: Metrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = Metrics()
    return _metrics
