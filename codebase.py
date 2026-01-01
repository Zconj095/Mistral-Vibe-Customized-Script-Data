from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

from pydantic import BaseModel, Field

from vibe.core.config import CodebaseConfig, VibeConfig


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

ENTRYPOINT_NAMES = {
    "main",
    "app",
    "server",
    "index",
    "cli",
    "__init__",
}

EXTENSION_HINTS = [
    ".py",
    ".pyi",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".kt",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".m",
    ".mm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
]

PY_IMPORT_RE = re.compile(r"^\s*import\s+(.+)")
PY_FROM_RE = re.compile(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+")

JS_IMPORT_FROM_RE = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
JS_IMPORT_SIDE_RE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]")
JS_EXPORT_FROM_RE = re.compile(r"^\s*export\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
JS_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)")
JS_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)")

C_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]")

JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(static\s+)?([A-Za-z0-9_\.]+)")
CSHARP_USING_RE = re.compile(r"^\s*using\s+([A-Za-z0-9_\.]+)")
SWIFT_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_\.]+)")

GO_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+"([^"]+)"')
GO_IMPORT_BLOCK_START_RE = re.compile(r"^\s*import\s*\(")
GO_IMPORT_SPEC_RE = re.compile(r'^\s*(?:[A-Za-z0-9_\.]+\s+)?\"([^\"]+)\"')

RUST_MOD_RE = re.compile(r"^\s*mod\s+([A-Za-z0-9_]+)\s*;")
RUST_USE_RE = re.compile(r"^\s*use\s+([A-Za-z0-9_:]+)")
RUST_EXTERN_CRATE_RE = re.compile(r"^\s*extern\s+crate\s+([A-Za-z0-9_]+)")

PHP_INCLUDE_RE = re.compile(
    r"^\s*(include|include_once|require|require_once)\s*[\"']([^\"']+)[\"']"
)
RUBY_REQUIRE_RE = re.compile(r"^\s*require(?:_relative)?\s+[\"']([^\"']+)[\"']")


@dataclass(frozen=True)
class RawReference:
    target: str
    relation: str


class CodebaseChunk(BaseModel):
    start_line: int
    end_line: int
    token_counts: dict[str, int]


class CodebaseEdge(BaseModel):
    target: str
    relation: str
    resolved: bool


class CodebaseFileEntry(BaseModel):
    path: str
    size: int
    mtime: float
    extension: str
    token_counts: dict[str, int]
    chunks: list[CodebaseChunk]
    edges: list[CodebaseEdge] = Field(default_factory=list)


class CodebaseIndex(BaseModel):
    version: int = 1
    root: str
    created_at: str
    total_files: int
    total_bytes: int
    truncated: bool
    token_df: dict[str, int]
    files: dict[str, CodebaseFileEntry]
    errors: list[str] = Field(default_factory=list)


class CodebaseIndexStats(BaseModel):
    total_files: int
    total_bytes: int
    truncated: bool
    created_at: str
    errors: list[str]


class CodebaseMatch(BaseModel):
    path: str
    score: float
    chunk: CodebaseChunk | None = None


class CodebaseManager:
    def __init__(self, config: VibeConfig) -> None:
        self.config = config
        self.root = config.effective_workdir.resolve()
        self._index_path = config.codebase.index_path
        self._index: CodebaseIndex | None = None

    def load_index(self) -> CodebaseIndex | None:
        if not self._index_path.exists():
            return None
        try:
            payload = json.loads(self._index_path.read_text("utf-8"))
            index = CodebaseIndex.model_validate(payload)
        except (OSError, json.JSONDecodeError):
            return None
        if Path(index.root).resolve() != self.root:
            return None
        return index

    def save_index(self, index: CodebaseIndex) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = index.model_dump(mode="json")
        self._index_path.write_text(json.dumps(payload, indent=2), "utf-8")

    def build_index(self) -> CodebaseIndex:
        config = self.config.codebase
        files = self._gather_files(self.root, config)
        token_df: dict[str, int] = {}
        entries: dict[str, CodebaseFileEntry] = {}
        total_bytes = 0
        truncated = False
        errors: list[str] = []
        references: dict[str, list[RawReference]] = {}

        for file_path in files:
            if len(entries) >= config.max_files:
                truncated = True
                break
            try:
                stat_info = file_path.stat()
                size = stat_info.st_size
                mtime = stat_info.st_mtime
            except OSError as exc:
                errors.append(f"{file_path}: {exc}")
                continue
            if size > config.max_file_bytes:
                continue
            if total_bytes + size > config.max_total_bytes:
                truncated = True
                break

            try:
                content = file_path.read_text("utf-8", errors="ignore")
            except OSError as exc:
                errors.append(f"{file_path}: {exc}")
                continue
            total_bytes += size
            rel_path = file_path.relative_to(self.root).as_posix()
            extension = file_path.suffix.lower()
            token_counts = _token_counts(content, config.min_token_length)
            token_counts = _trim_counts(token_counts, config.max_tokens_per_file)
            for token in token_counts:
                token_df[token] = token_df.get(token, 0) + 1

            lines = content.splitlines()
            chunks = _build_chunks(
                lines,
                config.chunk_lines,
                config.max_chunks_per_file,
                config.min_token_length,
                config.max_tokens_per_chunk,
            )
            refs = _extract_references(extension, lines)
            references[rel_path] = refs

            entries[rel_path] = CodebaseFileEntry(
                path=rel_path,
                size=size,
                mtime=mtime,
                extension=extension,
                token_counts=token_counts,
                chunks=chunks,
                edges=[],
            )

        index = CodebaseIndex(
            root=str(self.root),
            created_at=datetime.now(timezone.utc).isoformat(),
            total_files=len(entries),
            total_bytes=total_bytes,
            truncated=truncated,
            token_df=token_df,
            files=entries,
            errors=errors,
        )
        _attach_edges(
            index,
            references,
            self.root,
            config.include_unresolved,
            config.max_edges_per_file,
        )
        self._index = index
        self.save_index(index)
        return index

    def ensure_index(self) -> CodebaseIndex | None:
        if self._index is not None:
            return self._index
        if index := self.load_index():
            self._index = index
            return index
        if not self.config.codebase.auto_index:
            return None
        return self.build_index()

    def build_index_stats(self) -> CodebaseIndexStats:
        index = self.build_index()
        return CodebaseIndexStats(
            total_files=index.total_files,
            total_bytes=index.total_bytes,
            truncated=index.truncated,
            created_at=index.created_at,
            errors=index.errors,
        )

    def get_index_stats(self) -> CodebaseIndexStats | None:
        index = self.ensure_index()
        if not index:
            return None
        return CodebaseIndexStats(
            total_files=index.total_files,
            total_bytes=index.total_bytes,
            truncated=index.truncated,
            created_at=index.created_at,
            errors=index.errors,
        )

    def search(self, query: str) -> list[CodebaseMatch]:
        index = self.ensure_index()
        if not index:
            return []
        tokens = _tokenize_query(query, self.config.codebase.min_token_length)
        if not tokens:
            return []
        idf = _build_idf(index)
        matches: list[CodebaseMatch] = []
        for entry in index.files.values():
            score = _score_tokens(tokens, entry.token_counts, idf)
            if score <= 0:
                continue
            chunk = _best_chunk(tokens, entry.chunks, idf)
            matches.append(
                CodebaseMatch(path=entry.path, score=score, chunk=chunk)
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: self.config.codebase.top_k_files]

    def build_context(self, query: str) -> str:
        matches = self.search(query)
        if not matches:
            return ""
        index = self._index
        if index is None:
            return ""
        max_chars = self.config.codebase.context_max_chars
        lines: list[str] = [
            "<CODEBASE_CONTEXT>",
            f"Query: {query.strip()}",
        ]
        used_chars = sum(len(line) + 1 for line in lines)
        for match in matches:
            entry = index.files.get(match.path)
            if entry is None or match.chunk is None:
                continue
            snippet = _read_chunk(self.root / match.path, match.chunk)
            if not snippet:
                continue
            header = (
                f"\n### {match.path} "
                f"(lines {match.chunk.start_line}-{match.chunk.end_line})"
            )
            block = _format_code_block(entry.extension, snippet)
            connections = _format_connections(entry.edges)
            addition = header + "\n" + block + connections
            if used_chars + len(addition) > max_chars:
                break
            lines.append(header)
            lines.append(block)
            if connections:
                lines.append(connections)
            used_chars += len(addition)

        lines.append("</CODEBASE_CONTEXT>")
        return "\n".join(lines)

    def build_summary_context(self) -> str:
        index = self.ensure_index()
        if not index:
            return ""
        summary_lines = _summarize_index(index, self.root, self.config.codebase)
        return "\n".join(summary_lines)

    def build_graph_summary(self) -> str:
        index = self.ensure_index()
        if not index:
            return "No codebase index available."
        return _summarize_graph(index)

    def _gather_files(self, root: Path, config: CodebaseConfig) -> list[Path]:
        include_globs = _normalize_globs(config.include_globs)
        exclude_globs = _normalize_globs(config.exclude_globs)
        if not include_globs:
            return []
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            if rel_dir == ".":
                rel_dir = ""
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_excluded(
                    _join_rel(rel_dir, name) + "/", exclude_globs
                )
            ]
            for filename in filenames:
                rel_file = _join_rel(rel_dir, filename)
                if _is_excluded(rel_file, exclude_globs):
                    continue
                if not _is_included(rel_file, include_globs):
                    continue
                files.append(Path(dirpath) / filename)
                if len(files) >= config.max_files:
                    return files
        return files


def _normalize_globs(globs: Iterable[str]) -> list[str]:
    return [value.strip() for value in globs if value and value.strip()]


def _is_excluded(rel_path: str, exclude_globs: list[str]) -> bool:
    if not exclude_globs:
        return False
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs)


def _is_included(rel_path: str, include_globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in include_globs)


def _join_rel(base: str, name: str) -> str:
    if not base:
        return name
    return f"{base}/{name}"


def _token_counts(text: str, min_len: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) < min_len:
            continue
        if lowered in STOPWORDS:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    return counts


def _trim_counts(counts: dict[str, int], limit: int) -> dict[str, int]:
    if limit <= 0:
        return {}
    if len(counts) <= limit:
        return counts
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:limit])


def _build_chunks(
    lines: list[str],
    chunk_lines: int,
    max_chunks: int,
    min_token_length: int,
    max_tokens_per_chunk: int,
) -> list[CodebaseChunk]:
    if chunk_lines <= 0:
        return []
    chunks: list[CodebaseChunk] = []
    for start in range(0, len(lines), chunk_lines):
        if len(chunks) >= max_chunks:
            break
        end = min(start + chunk_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        counts = _token_counts(chunk_text, min_token_length)
        counts = _trim_counts(counts, max_tokens_per_chunk)
        chunks.append(
            CodebaseChunk(
                start_line=start + 1,
                end_line=end,
                token_counts=counts,
            )
        )
    return chunks


def _tokenize_query(text: str, min_len: int) -> list[str]:
    tokens = _token_counts(text, min_len)
    return list(tokens.keys())


def _build_idf(index: CodebaseIndex) -> dict[str, float]:
    doc_count = max(index.total_files, 1)
    return {
        token: math.log((doc_count + 1) / (df + 1)) + 1.0
        for token, df in index.token_df.items()
    }


def _score_tokens(
    query_tokens: list[str],
    token_counts: dict[str, int],
    idf: dict[str, float],
) -> float:
    total = sum(token_counts.values())
    if total <= 0:
        return 0.0
    score = 0.0
    for token in query_tokens:
        count = token_counts.get(token, 0)
        if count <= 0:
            continue
        score += (count / total) * idf.get(token, 0.0)
    return score


def _best_chunk(
    query_tokens: list[str],
    chunks: list[CodebaseChunk],
    idf: dict[str, float],
) -> CodebaseChunk | None:
    if not chunks:
        return None
    best = max(
        chunks,
        key=lambda chunk: _score_tokens(query_tokens, chunk.token_counts, idf),
    )
    if _score_tokens(query_tokens, best.token_counts, idf) <= 0:
        return None
    return best


def _read_chunk(path: Path, chunk: CodebaseChunk) -> str:
    try:
        lines = path.read_text("utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    start = max(chunk.start_line - 1, 0)
    end = min(chunk.end_line, len(lines))
    if start >= end:
        return ""
    return "\n".join(lines[start:end])


def _format_code_block(extension: str, content: str) -> str:
    lang = extension.lstrip(".") or "text"
    return f"```{lang}\n{content}\n```"


def _format_connections(edges: list[CodebaseEdge]) -> str:
    if not edges:
        return ""
    lines = ["Connections:"]
    for edge in edges[:5]:
        target = edge.target
        suffix = "" if edge.resolved else " (external)"
        lines.append(f"- {edge.relation}: {target}{suffix}")
    return "\n".join(lines)


def _extract_references(extension: str, lines: list[str]) -> list[RawReference]:
    match extension:
        case ".py" | ".pyi":
            return _extract_python_refs(lines)
        case ".js" | ".jsx" | ".ts" | ".tsx" | ".mjs" | ".cjs":
            return _extract_js_refs(lines)
        case ".java" | ".kt":
            return _extract_java_refs(lines)
        case ".cs":
            return _extract_csharp_refs(lines)
        case ".go":
            return _extract_go_refs(lines)
        case ".rs":
            return _extract_rust_refs(lines)
        case ".c" | ".cpp" | ".h" | ".hpp" | ".m" | ".mm":
            return _extract_c_refs(lines)
        case ".php":
            return _extract_php_refs(lines)
        case ".rb":
            return _extract_ruby_refs(lines)
        case ".swift":
            return _extract_swift_refs(lines)
        case _:
            return []


def _extract_python_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if match := PY_FROM_RE.match(line):
            module = match.group(1).strip()
            if module:
                refs.append(RawReference(module, "import"))
            continue
        if match := PY_IMPORT_RE.match(line):
            modules = _split_python_imports(match.group(1))
            refs.extend(RawReference(module, "import") for module in modules)
    return refs


def _split_python_imports(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    modules: list[str] = []
    for part in parts:
        if " as " in part:
            part = part.split(" as ")[0].strip()
        modules.append(part)
    return modules


def _extract_js_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if match := JS_IMPORT_FROM_RE.match(line):
            refs.append(RawReference(match.group(1), "import"))
            continue
        if match := JS_IMPORT_SIDE_RE.match(line):
            refs.append(RawReference(match.group(1), "import"))
        if match := JS_EXPORT_FROM_RE.match(line):
            refs.append(RawReference(match.group(1), "export"))
        for target in JS_REQUIRE_RE.findall(line):
            refs.append(RawReference(target, "require"))
        for target in JS_DYNAMIC_IMPORT_RE.findall(line):
            refs.append(RawReference(target, "import"))
    return refs


def _extract_java_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := JAVA_IMPORT_RE.match(line):
            module = match.group(2)
            if module:
                refs.append(RawReference(module, "import"))
    return refs


def _extract_csharp_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if " using " in f" {line} " and "=" in line:
            continue
        if " static " in f" {line} ":
            continue
        if match := CSHARP_USING_RE.match(line):
            refs.append(RawReference(match.group(1), "using"))
    return refs


def _extract_swift_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := SWIFT_IMPORT_RE.match(line):
            refs.append(RawReference(match.group(1), "import"))
    return refs


def _extract_go_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    in_block = False
    for line in lines:
        if in_block:
            if line.strip().startswith(")"):
                in_block = False
                continue
            if match := GO_IMPORT_SPEC_RE.match(line):
                refs.append(RawReference(match.group(1), "import"))
            continue
        if match := GO_IMPORT_SINGLE_RE.match(line):
            refs.append(RawReference(match.group(1), "import"))
            continue
        if GO_IMPORT_BLOCK_START_RE.match(line):
            in_block = True
    return refs


def _extract_rust_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := RUST_MOD_RE.match(line):
            refs.append(RawReference(match.group(1), "mod"))
            continue
        if match := RUST_USE_RE.match(line):
            refs.append(RawReference(match.group(1), "use"))
            continue
        if match := RUST_EXTERN_CRATE_RE.match(line):
            refs.append(RawReference(match.group(1), "extern"))
    return refs


def _extract_c_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := C_INCLUDE_RE.match(line):
            refs.append(RawReference(match.group(1), "include"))
    return refs


def _extract_php_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := PHP_INCLUDE_RE.match(line):
            refs.append(RawReference(match.group(2), "include"))
    return refs


def _extract_ruby_refs(lines: list[str]) -> list[RawReference]:
    refs: list[RawReference] = []
    for line in lines:
        if match := RUBY_REQUIRE_RE.match(line):
            refs.append(RawReference(match.group(1), "require"))
    return refs


class _Resolver:
    def __init__(self, root: Path, files: dict[str, CodebaseFileEntry]) -> None:
        self.root = root
        self.files = files
        self.module_map: dict[str, str] = {}
        self.no_ext_map: dict[str, str] = {}
        self.stem_map: dict[str, list[str]] = {}
        self._index_files()

    def _index_files(self) -> None:
        for rel_path in self.files:
            path = Path(rel_path)
            stem = path.stem
            module = _module_name(rel_path)
            no_ext = path.with_suffix("").as_posix()
            self.module_map[module] = rel_path
            self.no_ext_map[no_ext] = rel_path
            self.stem_map.setdefault(stem, []).append(rel_path)
            if stem in {"__init__", "index"}:
                module_dir = path.parent.as_posix()
                if module_dir and module_dir != ".":
                    self.module_map[module_dir.replace("/", ".")] = rel_path

    def resolve(self, target: str, source_rel: str) -> list[str]:
        if _is_path_like(target):
            return self._resolve_path(target, source_rel)
        return self._resolve_module(target, source_rel)

    def _resolve_path(self, target: str, source_rel: str) -> list[str]:
        normalized = target.replace("\\", "/")
        if normalized.startswith("@/"):
            normalized = normalized[2:]
        base_dir = Path(source_rel).parent
        results: list[str] = []
        candidates = _path_candidates(normalized, base_dir)
        for candidate in candidates:
            if rel := _resolve_rel(self.root, candidate):
                if rel in self.files:
                    results.append(rel)
                    continue
                if rel in self.no_ext_map:
                    results.append(self.no_ext_map[rel])
        return _unique(results)

    def _resolve_module(self, target: str, source_rel: str) -> list[str]:
        module = target.strip().strip(";")
        module = module.replace("::", ".")
        if module.endswith(".*"):
            module = module[:-2]
        if module.startswith("."):
            module = _resolve_relative_module(_module_name(source_rel), module) or module
            module = module.lstrip(".")
        results: list[str] = []
        if module in self.module_map:
            results.append(self.module_map[module])
        if module.replace(".", "/") in self.no_ext_map:
            results.append(self.no_ext_map[module.replace(".", "/")])
        tail = module.split(".")[-1] if module else ""
        if tail and tail in self.stem_map and len(self.stem_map[tail]) == 1:
            results.append(self.stem_map[tail][0])
        return _unique(results)


def _path_candidates(target: str, base_dir: Path) -> list[Path]:
    path = Path(target)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(base_dir / path)
    if path.suffix:
        return candidates
    for ext in EXTENSION_HINTS:
        candidates.append(candidates[0].with_suffix(ext))
        candidates.append(candidates[0] / f"index{ext}")
    candidates.append(candidates[0] / "__init__.py")
    return candidates


def _resolve_rel(root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _module_name(rel_path: str) -> str:
    path = Path(rel_path)
    no_ext = path.with_suffix("").as_posix()
    if path.stem in {"__init__", "index"}:
        parent = path.parent.as_posix()
        return parent.replace("/", ".") if parent and parent != "." else ""
    return no_ext.replace("/", ".")


def _resolve_relative_module(base: str | None, ref: str) -> str | None:
    if not base:
        return None
    dots = len(ref) - len(ref.lstrip("."))
    remainder = ref[dots:]
    parts = base.split(".") if base else []
    if dots > len(parts):
        return remainder or None
    prefix = parts[: len(parts) - dots]
    if remainder:
        return ".".join(prefix + remainder.split("."))
    return ".".join(prefix)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _is_path_like(target: str) -> bool:
    if target.startswith((".", "/", "\\")):
        return True
    if "/" in target or "\\" in target:
        return True
    return any(target.lower().endswith(ext) for ext in EXTENSION_HINTS)


def _attach_edges(
    index: CodebaseIndex,
    references: dict[str, list[RawReference]],
    root: Path,
    include_unresolved: bool,
    max_edges_per_file: int,
) -> None:
    resolver = _Resolver(root, index.files)
    for rel_path, refs in references.items():
        entry = index.files.get(rel_path)
        if entry is None:
            continue
        edges: list[CodebaseEdge] = []
        for ref in refs:
            targets = resolver.resolve(ref.target, rel_path)
            if targets:
                edges.extend(
                    CodebaseEdge(
                        target=target, relation=ref.relation, resolved=True
                    )
                    for target in targets
                )
            elif include_unresolved:
                edges.append(
                    CodebaseEdge(
                        target=ref.target, relation=ref.relation, resolved=False
                    )
                )
            if len(edges) >= max_edges_per_file:
                break
        entry.edges = edges[:max_edges_per_file]


def _summarize_index(
    index: CodebaseIndex,
    root: Path,
    config: CodebaseConfig,
) -> list[str]:
    ext_counts: dict[str, int] = {}
    dir_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    entrypoints: list[str] = []

    for rel_path, entry in index.files.items():
        ext = entry.extension or ""
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        parent = Path(rel_path).parent.as_posix()
        if parent and parent != ".":
            dir_counts[parent] = dir_counts.get(parent, 0) + 1
        stem = Path(rel_path).stem.lower()
        if stem in ENTRYPOINT_NAMES:
            entrypoints.append(rel_path)
        for token, count in entry.token_counts.items():
            token_counts[token] = token_counts.get(token, 0) + count

    lines = [
        "<CODEBASE_INDEX>",
        f"Root: {root}",
        f"Files indexed: {index.total_files}",
        f"Total bytes: {index.total_bytes}",
    ]
    if index.truncated:
        lines.append("Note: Indexing truncated due to limits.")

    lines.append("Top file types:")
    for ext, count in _top_items(ext_counts, 8):
        label = ext if ext else "(no extension)"
        lines.append(f"- {label}: {count}")

    lines.append("Top directories:")
    for name, count in _top_items(dir_counts, 8):
        lines.append(f"- {name}: {count}")

    if entrypoints:
        lines.append("Entry-point candidates:")
        for path in entrypoints[:10]:
            lines.append(f"- {path}")

    lines.append("Top terms:")
    for token, count in _top_items(token_counts, 12):
        lines.append(f"- {token}: {count}")

    lines.extend(_preview_key_files(root, entrypoints, config))
    lines.append("</CODEBASE_INDEX>")
    return lines


def _preview_key_files(
    root: Path, entrypoints: list[str], config: CodebaseConfig
) -> list[str]:
    if config.summary_max_files <= 0 or config.summary_max_chars <= 0:
        return []
    previews: list[str] = []
    used = 0
    for rel_path in entrypoints[: config.summary_max_files]:
        path = root / rel_path
        if not path.exists():
            continue
        try:
            content = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        snippet = "\n".join(content.splitlines()[:40])
        block = f"\n### Preview: {rel_path}\n{_format_code_block(path.suffix, snippet)}"
        if used + len(block) > config.summary_max_chars:
            break
        previews.append(block)
        used += len(block)
    return previews


def _top_items(values: dict[str, int], limit: int) -> list[tuple[str, int]]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]


def _summarize_graph(index: CodebaseIndex) -> str:
    if not index.files:
        return "No codebase entries available."
    edges: list[tuple[str, CodebaseEdge]] = []
    for path, entry in index.files.items():
        for edge in entry.edges:
            edges.append((path, edge))
    total_edges = len(edges)
    degree: dict[str, int] = {}
    for src, edge in edges:
        degree[src] = degree.get(src, 0) + 1
        if edge.resolved:
            degree[edge.target] = degree.get(edge.target, 0) + 1
    top_nodes = sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:10]

    lines = [
        "Codebase dependency graph:",
        f"- Files: {len(index.files)}",
        f"- Edges: {total_edges}",
        "Top hubs:",
    ]
    for node, count in top_nodes:
        lines.append(f"- {node}: {count}")
    if edges:
        lines.append("Sample edges:")
    for src, edge in edges[:20]:
        target = edge.target
        suffix = "" if edge.resolved else " (external)"
        lines.append(f"- {src} -> {target} [{edge.relation}]{suffix}")
    return "\n".join(lines)
