"""
File Handling Mixin for PAL MCP Tools

Handles file path validation, conversation-aware file deduplication,
prompt.txt handling, and token-aware file embedding.
Extracted from BaseTool to separate concerns.
"""

import logging
import os
from typing import Any, Optional

from utils.conversation_memory import get_conversation_file_list, get_thread
from utils.file_utils import read_file_content, read_files

logger = logging.getLogger(__name__)


class FileHandlingMixin:
    """Mixin providing file handling, validation, and embedding for tools."""

    def validate_file_paths(self, request) -> Optional[str]:
        """Validate that all file paths in the request are absolute.

        Returns error message if validation fails, None if all paths are valid.
        """
        file_fields = [
            "absolute_file_paths",
            "file",
            "path",
            "directory",
            "notebooks",
            "test_examples",
            "style_guide_examples",
            "files_checked",
            "relevant_files",
        ]

        for field_name in file_fields:
            if hasattr(request, field_name):
                field_value = getattr(request, field_name)
                if field_value is None:
                    continue

                paths_to_check = field_value if isinstance(field_value, list) else [field_value]

                for path in paths_to_check:
                    if path and not os.path.isabs(path):
                        return f"All file paths must be FULL absolute paths. Invalid path: '{path}'"

        return None

    def get_conversation_embedded_files(self, continuation_id: Optional[str]) -> list[str]:
        """Get list of files already embedded in conversation history."""
        if not continuation_id:
            return []

        thread_context = get_thread(continuation_id)
        if not thread_context:
            return []

        embedded_files = get_conversation_file_list(thread_context)
        logger.debug(f"[FILES] {self.name}: Found {len(embedded_files)} embedded files")
        return embedded_files

    def filter_new_files(self, requested_files: list[str], continuation_id: Optional[str]) -> list[str]:
        """Filter out files already embedded in conversation history."""
        logger.debug(f"[FILES] {self.name}: Filtering {len(requested_files)} requested files")

        if not continuation_id:
            logger.debug(f"[FILES] {self.name}: New conversation, all {len(requested_files)} files are new")
            return requested_files

        try:
            embedded_files = set(self.get_conversation_embedded_files(continuation_id))
            logger.debug(f"[FILES] {self.name}: Found {len(embedded_files)} embedded files in conversation")

            if not embedded_files:
                logger.debug(f"{self.name} tool: No files found in conversation history for thread {continuation_id}")
                return requested_files

            new_files = [f for f in requested_files if f not in embedded_files]
            logger.debug(
                f"[FILES] {self.name}: After filtering: {len(new_files)} new files, "
                f"{len(requested_files) - len(new_files)} already embedded"
            )

            if len(new_files) < len(requested_files):
                skipped = [f for f in requested_files if f in embedded_files]
                logger.debug(
                    f"{self.name} tool: Filtering {len(skipped)} files already in conversation history: {', '.join(skipped)}"
                )

            return new_files

        except Exception as e:
            logger.warning(f"{self.name} tool: Error checking conversation history for {continuation_id}: {e}")
            logger.warning(f"{self.name} tool: Including all requested files as fallback")
            return requested_files

    def handle_prompt_file(self, files: Optional[list[str]]) -> tuple[Optional[str], Optional[list[str]]]:
        """Check for and handle prompt.txt in the absolute file paths list.

        If prompt.txt is found, reads its content and removes it from the files list.
        This mechanism works around MCP's ~25K token limit.
        """
        if not files:
            return None, files

        prompt_content = None
        updated_files = []

        for file_path in files:
            if os.path.basename(file_path) == "prompt.txt":
                try:
                    content, _ = read_file_content(file_path)
                    if "--- BEGIN FILE:" in content and "--- END FILE:" in content:
                        lines = content.split("\n")
                        in_content = False
                        content_lines = []
                        for line in lines:
                            if line.startswith("--- BEGIN FILE:"):
                                in_content = True
                                continue
                            elif line.startswith("--- END FILE:"):
                                break
                            elif in_content:
                                content_lines.append(line)
                        prompt_content = "\n".join(content_lines)
                    else:
                        if not content.startswith("\n--- ERROR"):
                            prompt_content = content
                        else:
                            prompt_content = None
                except Exception:
                    pass
            else:
                updated_files.append(file_path)

        return prompt_content, updated_files if updated_files else None

    def _prepare_file_content_for_prompt(
        self,
        request_files: list[str],
        continuation_id: Optional[str],
        context_description: str = "New files",
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1_000,
        remaining_budget: Optional[int] = None,
        arguments: Optional[dict] = None,
        model_context: Optional[Any] = None,
    ) -> tuple[str, list[str]]:
        """Centralized file processing implementing dual prioritization strategy.

        Returns (formatted_file_content, actually_processed_files).
        """
        if not request_files:
            return "", []

        # Extract remaining budget from arguments if available
        if remaining_budget is None:
            args_to_use = arguments or getattr(self, "_current_arguments", {})
            remaining_budget = args_to_use.get("_remaining_tokens")

        # Determine effective token budget
        if remaining_budget is not None:
            effective_max_tokens = remaining_budget - reserve_tokens
        elif max_tokens is not None:
            effective_max_tokens = max_tokens - reserve_tokens
        else:
            if not model_context:
                model_context = getattr(self, "_model_context", None)
                if not model_context:
                    logger.error(f"[FILES] {self.name}: _prepare_file_content_for_prompt called without model_context.")
                    raise RuntimeError("Model context not provided for file preparation.")

            try:
                token_allocation = model_context.calculate_token_allocation()
                effective_max_tokens = token_allocation.file_tokens - reserve_tokens
                logger.debug(
                    f"[FILES] {self.name}: Using model context for {model_context.model_name}: "
                    f"{token_allocation.file_tokens:,} file tokens from {token_allocation.total_tokens:,} total"
                )
            except Exception as e:
                logger.error(f"[FILES] {self.name}: Failed to calculate token allocation: {e}", exc_info=True)
                effective_max_tokens = 100_000 - reserve_tokens

        effective_max_tokens = max(1000, effective_max_tokens)

        files_to_embed = self.filter_new_files(request_files, continuation_id)
        logger.debug(f"[FILES] {self.name}: Will embed {len(files_to_embed)} files after filtering")

        if files_to_embed:
            logger.info(
                f"[FILE_PROCESSING] {self.name} tool will embed new files: "
                f"{', '.join([os.path.basename(f) for f in files_to_embed])}"
            )
        else:
            logger.info(f"[FILE_PROCESSING] {self.name} tool: No new files to embed (all in conversation history)")

        content_parts = []
        actually_processed_files = []

        if files_to_embed:
            logger.debug(f"{self.name} tool embedding {len(files_to_embed)} new files: {', '.join(files_to_embed)}")
            try:
                from utils.file_utils import expand_paths

                expanded_files = expand_paths(files_to_embed)
                logger.debug(
                    f"[FILES] {self.name}: Expanded {len(files_to_embed)} paths to {len(expanded_files)} individual files"
                )

                file_content = read_files(
                    files_to_embed,
                    max_tokens=effective_max_tokens + reserve_tokens,
                    reserve_tokens=reserve_tokens,
                    include_line_numbers=self.wants_line_numbers_by_default(),
                )
                content_parts.append(file_content)
                actually_processed_files.extend(expanded_files)

                from utils.token_utils import estimate_tokens

                content_tokens = estimate_tokens(file_content)
                logger.debug(
                    f"{self.name} tool successfully embedded {len(files_to_embed)} files ({content_tokens:,} tokens)"
                )
            except Exception as e:
                logger.error(f"{self.name} tool failed to embed files {files_to_embed}: {type(e).__name__}: {e}")
                raise

        # Generate note about files already in conversation history
        if continuation_id and len(files_to_embed) < len(request_files):
            embedded_files = self.get_conversation_embedded_files(continuation_id)
            skipped_files = [f for f in request_files if f in embedded_files]
            if skipped_files:
                logger.debug(f"{self.name} tool skipping {len(skipped_files)} files already in conversation history")
                if content_parts:
                    content_parts.append("\n\n")
                note_lines = [
                    "--- NOTE: Additional files referenced in conversation history ---",
                    "The following files are already available in our conversation context:",
                    "\n".join(f"  - {f}" for f in skipped_files),
                    "--- END NOTE ---",
                ]
                content_parts.append("\n".join(note_lines))

        result = "".join(content_parts) if content_parts else ""
        logger.debug(
            f"[FILES] {self.name}: _prepare_file_content_for_prompt returning "
            f"{len(result)} chars, {len(actually_processed_files)} processed files"
        )
        return result, actually_processed_files
