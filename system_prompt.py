from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from vibe.core.config import PROJECT_DOC_FILENAMES
from vibe.core.config_path import INSTRUCTIONS_FILE
from vibe.core.llm.format import get_active_tool_classes
from vibe.core.prompts import UtilityPrompt
from vibe.core.utils import is_dangerous_directory, is_windows

if TYPE_CHECKING:
    from vibe.core.config import ProjectContextConfig, VibeConfig
    from vibe.core.tools.manager import ToolManager


def _load_user_instructions() -> str:
    try:
        return INSTRUCTIONS_FILE.path.read_text("utf-8", errors="ignore")
    except (FileNotFoundError, OSError):
        return ""


def _load_project_doc(workdir: Path, max_bytes: int) -> str:
    for name in PROJECT_DOC_FILENAMES:
        path = workdir / name
        try:
            content = path.read_text("utf-8", errors="ignore")
            if max_bytes > 0:
                return content[:max_bytes]
            return content
        except (FileNotFoundError, OSError):
            continue
    return ""


class ProjectContextProvider:
    def __init__(
        self, config: ProjectContextConfig, root_path: str | Path = "."
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.config = config
        self.gitignore_patterns = self._load_gitignore_patterns()
        self._file_count = 0
        self._start_time = 0.0

    def _load_gitignore_patterns(self) -> list[str]:
        gitignore_path = self.root_path / ".gitignore"
        patterns = []

        if gitignore_path.exists():
            try:
                patterns.extend(
                    line.strip()
                    for line in gitignore_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                )
            except Exception as e:
                print(f"Warning: Could not read .gitignore: {e}", file=sys.stderr)

        default_patterns = [
            ".git",
            ".git/*",
            "*.pyc",
            "__pycache__",
            "node_modules",
            "node_modules/*",
            ".env",
            ".DS_Store",
            "*.log",
            ".vscode/settings.json",
            ".idea/*",
            "dist",
            "build",
            "target",
            ".next",
            ".nuxt",
            "coverage",
            ".nyc_output",
            "*.egg-info",
            ".pytest_cache",
            ".tox",
            "vendor",
            "third_party",
            "deps",
            "*.min.js",
            "*.min.css",
            "*.bundle.js",
            "*.chunk.js",
            ".cache",
            "tmp",
            "temp",
            "logs",
        ]

        return patterns + default_patterns

    def _is_ignored(self, path: Path) -> bool:
        try:
            relative_path = path.relative_to(self.root_path)
            path_str = str(relative_path)

            for pattern in self.gitignore_patterns:
                if pattern.endswith("/"):
                    if path.is_dir() and fnmatch.fnmatch(f"{path_str}/", pattern):
                        return True
                elif fnmatch.fnmatch(path_str, pattern):
                    return True
                elif "*" in pattern or "?" in pattern:
                    if fnmatch.fnmatch(path_str, pattern):
                        return True

            return False
        except (ValueError, OSError):
            return True

    def _should_stop(self) -> bool:
        max_files = self.config.max_files
        timeout_seconds = self.config.timeout_seconds

        file_limit_reached = max_files > 0 and self._file_count >= max_files
        timeout_reached = (
            timeout_seconds > 0 and (time.time() - self._start_time) > timeout_seconds
        )
        return file_limit_reached or timeout_reached

    def _build_tree_structure_iterative(self) -> Generator[str]:
        self._start_time = time.time()
        self._file_count = 0

        yield from self._process_directory(self.root_path, "", 0, is_root=True)

    def _process_directory(
        self, path: Path, prefix: str, depth: int, is_root: bool = False
    ) -> Generator[str]:
        if depth > self.config.max_depth or self._should_stop():
            return

        try:
            all_items = list(path.iterdir())
            items = [item for item in all_items if not self._is_ignored(item)]

            items.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

            max_dirs = self.config.max_dirs_per_level
            show_truncation = max_dirs > 0 and len(items) > max_dirs
            if show_truncation:
                items = items[:max_dirs]

            for i, item in enumerate(items):
                if self._should_stop():
                    break

                is_last = i == len(items) - 1 and not show_truncation
                connector = "└── " if is_last else "├── "
                name = f"{item.name}{'/' if item.is_dir() else ''}"

                yield f"{prefix}{connector}{name}"
                self._file_count += 1

                if item.is_dir() and depth < self.config.max_depth:
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    yield from self._process_directory(item, child_prefix, depth + 1)

            if show_truncation and not self._should_stop():
                remaining = len(all_items) - len(items)
                yield f"{prefix}└── ... ({remaining} more items)"

        except (PermissionError, OSError):
            pass

    def get_directory_structure(self) -> str:
        lines = []
        max_items = self.config.max_files
        max_items_label = "no limit" if max_items <= 0 else str(max_items)
        header = (
            f"Directory structure of {self.root_path.name} "
            f"(depth={self.config.max_depth}, max {max_items_label} items):\n"
        )

        try:
            for line in self._build_tree_structure_iterative():
                lines.append(line)

                current_text = header + "\n".join(lines)
                if self.config.max_chars > 0 and (
                    len(current_text)
                    > self.config.max_chars - self.config.truncation_buffer
                ):
                    break

        except Exception as e:
            lines.append(f"Error building structure: {e}")

        structure = header + "\n".join(lines)

        if self.config.max_files > 0 and self._file_count >= self.config.max_files:
            structure += f"\n... (truncated at {self.config.max_files} files limit)"
        elif self.config.timeout_seconds > 0 and (
            time.time() - self._start_time
        ) > self.config.timeout_seconds:
            structure += (
                f"\n... (truncated due to {self.config.timeout_seconds}s timeout)"
            )
        elif self.config.max_chars > 0 and len(structure) > self.config.max_chars:
            structure += f"\n... (truncated at {self.config.max_chars} characters)"

        return structure
    def get_git_status(self) -> str:
        try:
            timeout = min(self.config.timeout_seconds, 10.0)
            num_commits = self.config.default_commit_count

            current_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            main_branch = "main"
            try:
                branches_output = subprocess.run(
                    ["git", "branch", "-r"],
                    capture_output=True,
                    check=True,
                    cwd=self.root_path,
                    stdin=subprocess.DEVNULL if is_windows() else None,
                    text=True,
                    timeout=timeout,
                ).stdout
                if "origin/master" in branches_output:
                    main_branch = "master"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

            status_output = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            if status_output:
                status_lines = status_output.splitlines()
                MAX_GIT_STATUS_SIZE = 50
                if len(status_lines) > MAX_GIT_STATUS_SIZE:
                    status = (
                        f"({len(status_lines)} changes - use 'git status' for details)"
                    )
                else:
                    status = f"({len(status_lines)} changes)"
            else:
                status = "(clean)"

            log_output = subprocess.run(
                ["git", "log", "--oneline", f"-{num_commits}", "--decorate"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            recent_commits = []
            for line in log_output.split("\n"):
                if not (line := line.strip()):
                    continue

                if " " in line:
                    commit_hash, commit_msg = line.split(" ", 1)
                    if (
                        "(" in commit_msg
                        and ")" in commit_msg
                        and (paren_index := commit_msg.rfind("(")) > 0
                    ):
                        commit_msg = commit_msg[:paren_index].strip()
                    recent_commits.append(f"{commit_hash} {commit_msg}")
                else:
                    recent_commits.append(line)

            git_info_parts = [
                f"Current branch: {current_branch}",
                f"Main branch (you will usually use this for PRs): {main_branch}",
                f"Status: {status}",
            ]

            if recent_commits:
                git_info_parts.append("Recent commits:")
                git_info_parts.extend(recent_commits)

            return "\n".join(git_info_parts)

        except subprocess.TimeoutExpired:
            return "Git operations timed out (large repository)"
        except subprocess.CalledProcessError:
            return "Not a git repository or git not available"
        except Exception as e:
            return f"Error getting git status: {e}"

    def get_full_context(self) -> str:
        structure = self.get_directory_structure()
        git_status = self.get_git_status()

        large_repo_warning = ""
        if self.config.max_chars > 0 and (
            len(structure) >= self.config.max_chars - self.config.truncation_buffer
        ):
            large_repo_warning = (
                f" Large repository detected - showing summary view with depth limit {self.config.max_depth}. "
                f"Use the LS tool (passing a specific path), Bash tool, and other tools to explore nested directories in detail."
            )

        template = UtilityPrompt.PROJECT_CONTEXT.read()
        return template.format(
            large_repo_warning=large_repo_warning,
            structure=structure,
            abs_path=self.root_path,
            git_status=git_status,
        )


def _get_platform_name() -> str:
    platform_names = {
        "win32": "Windows",
        "darwin": "macOS",
        "linux": "Linux",
        "freebsd": "FreeBSD",
        "openbsd": "OpenBSD",
        "netbsd": "NetBSD",
    }
    return platform_names.get(sys.platform, "Unix-like")


def _get_default_shell() -> str:
    """Get the default shell used by asyncio.create_subprocess_shell.

    On Unix, this is always 'sh'.
    On Windows, this is COMSPEC or cmd.exe.
    """
    if is_windows():
        return os.environ.get("COMSPEC", "cmd.exe")
    return "sh"


def _get_os_system_prompt() -> str:
    shell = _get_default_shell()
    platform_name = _get_platform_name()
    prompt = f"The operating system is {platform_name} with shell `{shell}`"

    if is_windows():
        prompt += "\n" + _get_windows_system_prompt()
    return prompt


def _get_windows_system_prompt() -> str:
    return (
        "### COMMAND COMPATIBILITY RULES (MUST FOLLOW):\n"
        "- DO NOT use Unix commands like `ls`, `grep`, `cat` - they won't work on Windows\n"
        "- Use: `dir` (Windows) for directory listings\n"
        "- Use: backslashes (\\\\) for paths\n"
        "- Check command availability with: `where command` (Windows)\n"
        "- Script shebang: Not applicable on Windows\n"
        "### ALWAYS verify commands work on the detected platform before suggesting them"
    )


def _add_commit_signature() -> str:
    return (
        "When you want to commit changes, you will always use the 'git commit' bash command.\n"
        "It will always be suffixed with a line telling it was generated by Mistral Vibe with the appropriate co-authoring information.\n"
        "The format you will always uses is the following heredoc.\n\n"
        "```bash\n"
        "git commit -m <Commit message here>\n\n"
        "Generated by Mistral Vibe.\n"
        "Co-Authored-By: Mistral Vibe <vibe@mistral.ai>\n"
        "```"
    )

RESPONSE_LENGTH_GUIDANCE = {
    "concise": "Be concise and avoid unnecessary detail.",
    "balanced": "Use a balanced level of detail without being verbose.",
    "detailed": "Provide thorough detail and helpful context.",
}

RESPONSE_TONE_GUIDANCE = {
    "neutral": "Use a neutral, matter-of-fact tone.",
    "friendly": "Use a friendly, supportive tone without being overly chatty.",
    "professional": "Use a professional, businesslike tone.",
    "direct": "Be direct and to the point.",
}

RESPONSE_TASK_STYLE_GUIDANCE = {
    "standard": "Use the format that best serves the task.",
    "answer_only": "Provide just the answer or result unless asked for more.",
    "step_by_step": "Provide concise step-by-step instructions when applicable.",
    "code_only": "When returning code, output only code with no explanation.",
}

REASONING_MODE_GUIDANCE = {
    "low": "Use minimal reasoning and prioritize speed; keep analysis brief.",
    "medium": "Use balanced reasoning with a few key checks or alternatives.",
    "high": "Use thorough reasoning with explicit verification steps.",
    "extra_high": "Use very thorough reasoning, cross-checking assumptions and edge cases.",
}


def _build_style_prompt(config: VibeConfig) -> str:
    length_key = config.response_length.value
    tone_key = config.response_tone.value
    task_key = config.response_task_style.value

    length_text = RESPONSE_LENGTH_GUIDANCE.get(length_key, "")
    tone_text = RESPONSE_TONE_GUIDANCE.get(tone_key, "")
    task_text = RESPONSE_TASK_STYLE_GUIDANCE.get(task_key, "")

    lines = [
        "### RESPONSE STYLE",
        "Follow these preferences unless the user requests otherwise.",
    ]
    if length_text:
        lines.append(f"- Length: {length_text}")
    if tone_text:
        lines.append(f"- Tone: {tone_text}")
    if task_text:
        lines.append(f"- Task style: {task_text}")

    if custom_prompt := config.style_prompt.strip():
        lines.append("Additional style instructions:")
        lines.append(custom_prompt)

    if model_prompt := config.get_model_style_prompt().strip():
        lines.append("Model-specific style instructions:")
        lines.append(model_prompt)

    return "\n".join(lines)


def _format_project_templates(template_presets: dict[str, str]) -> str:
    if not template_presets:
        return "None configured."

    lines: list[str] = []
    for name, description in sorted(template_presets.items()):
        desc = description.strip()
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _build_project_mode_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.PROJECT_MODE.read()
    templates = _format_project_templates(config.project_mode.template_presets)
    return template.format(template_presets=templates)


@dataclass(frozen=True)
class LibraryLoadResult:
    text: str
    files_loaded: int
    total_bytes: int
    truncated: bool
    root: Path


def _iter_library_files(
    root: Path, include_globs: list[str], exclude_globs: list[str]
) -> list[Path]:
    if not root.exists():
        return []

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if include_globs and not any(
            fnmatch.fnmatch(rel_path, pattern) for pattern in include_globs
        ):
            continue
        if exclude_globs and any(
            fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs
        ):
            continue
        candidates.append(path)

    return sorted(candidates, key=lambda p: p.as_posix().lower())


def _truncate_text_bytes(text: str, max_bytes: int) -> tuple[str, int, bool]:
    if max_bytes <= 0:
        data = text.encode("utf-8")
        return text, len(data), False

    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text, len(data), False

    truncated = data[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, max_bytes, True


def _load_mistral_libraries(config: VibeConfig) -> LibraryLoadResult:
    mi_config = config.mistral_intelligence
    root = mi_config.library_dir
    include_globs = mi_config.include_globs
    exclude_globs = mi_config.exclude_globs

    files = _iter_library_files(root, include_globs, exclude_globs)
    max_files = mi_config.max_library_files
    max_file_bytes = mi_config.max_library_bytes
    max_total_bytes = mi_config.max_total_library_bytes

    sections: list[str] = []
    total_bytes = 0
    truncated = False
    loaded = 0

    for path in files:
        if max_files > 0 and loaded >= max_files:
            truncated = True
            break

        try:
            content = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue

        content, content_bytes, content_truncated = _truncate_text_bytes(
            content, max_file_bytes
        )
        if content_truncated:
            truncated = True

        if max_total_bytes > 0 and total_bytes + content_bytes > max_total_bytes:
            remaining = max_total_bytes - total_bytes
            if remaining <= 0:
                truncated = True
                break
            content, content_bytes, _ = _truncate_text_bytes(content, remaining)
            truncated = True

        if not content.strip():
            continue

        rel_path = path.relative_to(root).as_posix()
        sections.append(f"### Library: {rel_path}")
        sections.append(content.strip())
        total_bytes += content_bytes
        loaded += 1

        if max_total_bytes > 0 and total_bytes >= max_total_bytes:
            truncated = True
            break

    text = "\n\n".join(sections).strip()
    return LibraryLoadResult(
        text=text,
        files_loaded=loaded,
        total_bytes=total_bytes,
        truncated=truncated,
        root=root,
    )


def _build_mistral_intelligence_prompt(config: VibeConfig) -> str:
    libraries = _load_mistral_libraries(config)
    template = UtilityPrompt.MISTRAL_INTELLIGENCE.read()
    library_context = libraries.text or "No libraries loaded."
    return template.format(
        library_dir=libraries.root,
        libraries_loaded=libraries.files_loaded,
        library_bytes=libraries.total_bytes,
        library_truncated="yes" if libraries.truncated else "no",
        library_context=library_context,
    )


def _build_extended_thinking_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.EXTENDED_THINKING.read()
    thinking = config.extended_thinking
    return template.format(
        reasoning_style=thinking.reasoning_style,
        show_steps="yes" if thinking.show_steps else "no",
        max_rationale_sentences=thinking.max_rationale_sentences,
    )


def _build_thought_mode_prompt(config: VibeConfig) -> str:
    return UtilityPrompt.THOUGHT_MODE.read()


def _build_chain_of_thought_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.CHAIN_OF_THOUGHT.read()
    cot = config.chain_of_thought
    return template.format(max_rationale_sentences=cot.max_rationale_sentences)


def _build_latex_mode_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.LATEX_MODE.read()
    latex_mode = config.latex_mode
    return template.format(
        preserve_source="yes" if latex_mode.preserve_source else "no",
        use_code_fences="yes" if latex_mode.use_code_fences else "no",
    )


def _build_visual_layout_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.VISUAL_LAYOUT.read()
    layout = config.visual_layout
    return template.format(
        layout_style=layout.layout_style,
        include_layout_map="yes" if layout.include_layout_map else "no",
        use_tables="yes" if layout.use_tables else "no",
        use_ascii_maps="yes" if layout.use_ascii_maps else "no",
        max_sections=layout.max_sections,
        max_items_per_section=layout.max_items_per_section,
    )


def _build_agent_full_access_prompt(config: VibeConfig) -> str:
    return UtilityPrompt.AGENT_FULL_ACCESS.read()


def _build_reasoning_mode_prompt(config: VibeConfig) -> str:
    template = UtilityPrompt.REASONING_MODE.read()
    mode = config.reasoning_mode.value
    guidance = REASONING_MODE_GUIDANCE.get(mode, "Use appropriate reasoning depth.")
    return template.format(mode=mode, guidance=guidance)


def get_universal_system_prompt(tool_manager: ToolManager, config: VibeConfig) -> str:
    sections = [config.system_prompt]

    if config.include_commit_signature:
        sections.append(_add_commit_signature())

    if config.include_model_info:
        sections.append(f"Your model name is: `{config.active_model}`")

    if style_prompt := _build_style_prompt(config):
        sections.append(style_prompt)

    sections.append(_build_reasoning_mode_prompt(config))

    if config.agent_full_access.enabled:
        sections.append(_build_agent_full_access_prompt(config))

    if config.project_mode.enabled:
        sections.append(_build_project_mode_prompt(config))

    if config.mistral_intelligence.enabled:
        sections.append(_build_mistral_intelligence_prompt(config))

    if config.extended_thinking.enabled:
        sections.append(_build_extended_thinking_prompt(config))

    if config.thought_mode.enabled:
        sections.append(_build_thought_mode_prompt(config))

    if config.chain_of_thought.enabled:
        sections.append(_build_chain_of_thought_prompt(config))

    if config.latex_mode.enabled:
        sections.append(_build_latex_mode_prompt(config))

    if config.visual_layout.enabled:
        sections.append(_build_visual_layout_prompt(config))

    if config.include_prompt_detail:
        sections.append(_get_os_system_prompt())
        tool_prompts = []
        active_tools = get_active_tool_classes(tool_manager, config)
        for tool_class in active_tools:
            if prompt := tool_class.get_tool_prompt():
                tool_prompts.append(prompt)
        if tool_prompts:
            sections.append("\n---\n".join(tool_prompts))

        user_instructions = config.instructions.strip() or _load_user_instructions()
        if user_instructions.strip():
            sections.append(user_instructions)

    if config.include_project_context:
        project_context = config.project_context
        if config.project_mode.enabled:
            project_context = project_context.model_copy()
            project_context.max_files = config.project_mode.max_project_files
            project_context.max_doc_bytes = config.project_mode.max_project_doc_bytes

        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            template = UtilityPrompt.DANGEROUS_DIRECTORY.read()
            context = template.format(
                reason=reason.lower(), abs_path=Path(".").resolve()
            )
        else:
            context = ProjectContextProvider(
                config=project_context, root_path=config.effective_workdir
            ).get_full_context()

        sections.append(context)

        project_doc = _load_project_doc(
            config.effective_workdir, project_context.max_doc_bytes
        )
        if project_doc.strip():
            sections.append(project_doc)

    return "\n\n".join(sections)
