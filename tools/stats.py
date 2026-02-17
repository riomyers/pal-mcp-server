"""
Stats Tool - Display PAL MCP Server runtime metrics

Shows per-tool call counts, latency, error rates, and token usage.
All metrics are collected since server start (resets on restart).
"""

import logging
from typing import Any, Optional

from mcp.types import TextContent

from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool
from utils.metrics import get_metrics

logger = logging.getLogger(__name__)


class StatsTool(BaseTool):
    """Tool for displaying PAL MCP Server runtime metrics and health."""

    def get_name(self) -> str:
        return "stats"

    def get_description(self) -> str:
        return "Show server runtime metrics: per-tool call counts, latency, error rates, and uptime."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reset": {
                    "type": "boolean",
                    "description": "Reset all metrics counters",
                    "default": False,
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def get_annotations(self) -> Optional[dict[str, Any]]:
        return {"readOnlyHint": True}

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return ToolRequest

    def requires_model(self) -> bool:
        return False

    async def prepare_prompt(self, request: ToolRequest) -> str:
        return ""

    def format_response(self, response: str, request: ToolRequest, model_info: dict = None) -> str:
        return response

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        metrics = get_metrics()

        if arguments.get("reset", False):
            metrics.reset()
            output = ToolOutput(
                status="success",
                content="Metrics have been reset.",
                content_type="text",
            )
            return [TextContent(type="text", text=output.model_dump_json())]

        summary = metrics.get_summary()

        lines = ["# PAL MCP Server Metrics\n"]
        lines.append(f"**Uptime**: {summary['uptime_human']} ({summary['uptime_seconds']:,}s)")
        lines.append(f"**Total Calls**: {summary['total_calls']:,}")
        lines.append(f"**Total Errors**: {summary['total_errors']:,}")
        lines.append("")

        if summary["tools"]:
            lines.append("## Per-Tool Breakdown\n")
            lines.append("| Tool | Calls | Errors | Error Rate | Avg Latency | Max Latency | Tokens In | Tokens Out |")
            lines.append("|------|-------|--------|------------|-------------|-------------|-----------|------------|")

            for name, data in summary["tools"].items():
                lines.append(
                    f"| {name} | {data['calls']:,} | {data['errors']:,} | {data['error_rate']} | "
                    f"{data['avg_latency_ms']:,}ms | {data['max_latency_ms']:,}ms | "
                    f"{data['total_tokens_in']:,} | {data['total_tokens_out']:,} |"
                )
        else:
            lines.append("*No tool calls recorded yet.*")

        content = "\n".join(lines)

        output = ToolOutput(
            status="success",
            content=content,
            content_type="text",
            metadata=summary,
        )
        return [TextContent(type="text", text=output.model_dump_json())]

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.FAST_RESPONSE
