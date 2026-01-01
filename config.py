from __future__ import annotations

from enum import StrEnum, auto
import os
from pathlib import Path
import re
import shlex
import tomllib
from typing import Annotated, Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
import tomli_w

from vibe.core.config_path import (
    AGENT_DIR,
    CONFIG_DIR,
    CONFIG_FILE,
    GLOBAL_ENV_FILE,
    PROMPT_DIR,
    SESSION_LOG_DIR,
)
from vibe.core.prompts import SystemPrompt
from vibe.core.tools.base import BaseToolConfig

PROJECT_DOC_FILENAMES = ["AGENTS.md", "VIBE.md", ".vibe.md"]


def load_api_keys_from_env() -> None:
    if GLOBAL_ENV_FILE.path.is_file():
        env_vars = dotenv_values(GLOBAL_ENV_FILE.path)
        for key, value in env_vars.items():
            if value:
                os.environ.setdefault(key, value)


class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_key: str, provider_name: str) -> None:
        super().__init__(
            f"Missing {env_key} environment variable for {provider_name} provider"
        )
        self.env_key = env_key
        self.provider_name = provider_name


class MissingPromptFileError(RuntimeError):
    def __init__(self, system_prompt_id: str, prompt_dir: str) -> None:
        super().__init__(
            f"Invalid system_prompt_id value: '{system_prompt_id}'. "
            f"Must be one of the available prompts ({', '.join(f'{p.name.lower()}' for p in SystemPrompt)}), "
            f"or correspond to a .md file in {prompt_dir}"
        )
        self.system_prompt_id = system_prompt_id
        self.prompt_dir = prompt_dir


class WrongBackendError(RuntimeError):
    def __init__(self, backend: Backend, is_mistral_api: bool) -> None:
        super().__init__(
            f"Wrong backend '{backend}' for {'' if is_mistral_api else 'non-'}"
            f"mistral API. Use '{Backend.MISTRAL}' for mistral API and '{Backend.GENERIC}' for others."
        )
        self.backend = backend
        self.is_mistral_api = is_mistral_api


class TomlFileSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self.toml_data = self._load_toml()

    def _load_toml(self) -> dict[str, Any]:
        file = CONFIG_FILE.path
        try:
            with file.open("rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(f"Invalid TOML in {file}: {e}") from e
        except OSError as e:
            raise RuntimeError(f"Cannot read {file}: {e}") from e

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return self.toml_data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self.toml_data


class ProjectContextConfig(BaseSettings):
    max_chars: int = 40_000
    default_commit_count: int = 5
    max_doc_bytes: int = 32 * 1024
    truncation_buffer: int = 1_000
    max_depth: int = 3
    max_files: int = 1000
    max_dirs_per_level: int = 20
    timeout_seconds: float = 2.0


class SessionLoggingConfig(BaseSettings):
    save_dir: str = ""
    session_prefix: str = "session"
    enabled: bool = True

    @field_validator("save_dir", mode="before")
    @classmethod
    def set_default_save_dir(cls, v: str) -> str:
        if not v:
            return str(SESSION_LOG_DIR.path)
        return v

    @field_validator("save_dir", mode="after")
    @classmethod
    def expand_save_dir(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


class CodebaseConfig(BaseSettings):
    enabled: bool = False
    auto_index: bool = True
    index_path: Path = Field(
        default=Path.home() / ".vibe" / "memory" / "codebase_index.json",
        description="Path to the persisted codebase index.",
    )
    include_globs: list[str] = Field(
        default_factory=lambda: [
            "**/*.py",
            "**/*.pyi",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.mjs",
            "**/*.cjs",
            "**/*.java",
            "**/*.kt",
            "**/*.cs",
            "**/*.cpp",
            "**/*.c",
            "**/*.h",
            "**/*.hpp",
            "**/*.go",
            "**/*.rs",
            "**/*.php",
            "**/*.rb",
            "**/*.swift",
            "**/*.m",
            "**/*.mm",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.md",
            "**/*.txt",
        ]
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.idea/**",
            "**/.vscode/**",
            "**/dist/**",
            "**/build/**",
            "**/target/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.cache/**",
            "**/__pycache__/**",
        ]
    )
    max_files: int = 2000
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 50_000_000
    chunk_lines: int = 200
    max_chunks_per_file: int = 20
    min_token_length: int = 3
    max_tokens_per_file: int = 2000
    max_tokens_per_chunk: int = 400
    top_k_files: int = 8
    context_max_chars: int = 12_000
    summary_max_chars: int = 12_000
    summary_max_files: int = 5
    include_unresolved: bool = True
    max_edges_per_file: int = 50

    @field_validator("index_path", mode="before")
    @classmethod
    def set_default_index_path(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "memory" / "codebase_index.json"
        return Path(v)

    @field_validator("index_path", mode="after")
    @classmethod
    def expand_index_path(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class ProjectModeConfig(BaseSettings):
    enabled: bool = False
    max_read_bytes: int = 0
    max_write_bytes: int = 0
    max_edit_bytes: int = 0
    max_project_files: int = 0
    max_project_doc_bytes: int = 0
    template_presets: dict[str, str] = Field(default_factory=dict)


class MistralIntelligenceConfig(BaseSettings):
    enabled: bool = False
    library_dir: Path = Field(
        default=Path.home() / ".vibe" / "libraries" / "mistral_intelligence",
        description="Directory containing Mistral intelligence library files.",
    )
    max_library_files: int = 200
    max_library_bytes: int = 50_000
    max_total_library_bytes: int = 200_000
    include_globs: list[str] = Field(
        default_factory=lambda: [
            "**/*.md",
            "**/*.txt",
        ]
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/__pycache__/**",
        ]
    )

    @field_validator("library_dir", mode="before")
    @classmethod
    def set_default_library_dir(cls, v: Path | str) -> Path:
        if isinstance(v, Path):
            return v
        if not v or not str(v).strip():
            return Path.home() / ".vibe" / "libraries" / "mistral_intelligence"
        return Path(v)

    @field_validator("library_dir", mode="after")
    @classmethod
    def expand_library_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class ExtendedThinkingConfig(BaseSettings):
    enabled: bool = False
    reasoning_style: str = "deliberate"
    show_steps: bool = False
    max_rationale_sentences: int = 3


class ThoughtModeConfig(BaseSettings):
    enabled: bool = False


class ChainOfThoughtConfig(BaseSettings):
    enabled: bool = False
    max_rationale_sentences: int = 3


class LatexModeConfig(BaseSettings):
    enabled: bool = False
    preserve_source: bool = True
    use_code_fences: bool = True


class VisualLayoutConfig(BaseSettings):
    enabled: bool = False
    layout_style: str = "cards"
    include_layout_map: bool = True
    use_tables: bool = True
    use_ascii_maps: bool = True
    max_sections: int = 6
    max_items_per_section: int = 8


class AgentFullAccessConfig(BaseSettings):
    enabled: bool = False


class Backend(StrEnum):
    MISTRAL = auto()
    GENERIC = auto()
    UNSLOTH = auto()


class ResponseLength(StrEnum):
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"


class ResponseTone(StrEnum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    DIRECT = "direct"


class ResponseTaskStyle(StrEnum):
    STANDARD = "standard"
    ANSWER_ONLY = "answer_only"
    STEP_BY_STEP = "step_by_step"
    CODE_ONLY = "code_only"


class ReasoningMode(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTRA_HIGH = "extra_high"


class ProviderConfig(BaseModel):
    name: str
    api_base: str
    api_key_env_var: str = ""
    api_style: str = "openai"
    backend: Backend = Backend.GENERIC


class _MCPBase(BaseModel):
    name: str = Field(description="Short alias used to prefix tool names")
    prompt: str | None = Field(
        default=None, description="Optional usage hint appended to tool descriptions"
    )

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", v)
        normalized = normalized.strip("_-")
        return normalized[:256]


class _MCPHttpFields(BaseModel):
    url: str = Field(description="Base URL of the MCP HTTP server")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional HTTP headers when using 'http' transport (e.g., Authorization or X-API-Key)."
        ),
    )
    api_key_env: str = Field(
        default="",
        description=(
            "Environment variable name containing an API token to send for HTTP transport."
        ),
    )
    api_key_header: str = Field(
        default="Authorization",
        description=(
            "HTTP header name to carry the token when 'api_key_env' is set (e.g., 'Authorization' or 'X-API-Key')."
        ),
    )
    api_key_format: str = Field(
        default="Bearer {token}",
        description=(
            "Format string for the header value when 'api_key_env' is set. Use '{token}' placeholder."
        ),
    )

    def http_headers(self) -> dict[str, str]:
        hdrs = dict(self.headers or {})
        env_var = (self.api_key_env or "").strip()
        if env_var and (token := os.getenv(env_var)):
            target = (self.api_key_header or "").strip() or "Authorization"
            if not any(h.lower() == target.lower() for h in hdrs):
                try:
                    value = (self.api_key_format or "{token}").format(token=token)
                except Exception:
                    value = token
                hdrs[target] = value
        return hdrs


class MCPHttp(_MCPBase, _MCPHttpFields):
    transport: Literal["http"]


class MCPStreamableHttp(_MCPBase, _MCPHttpFields):
    transport: Literal["streamable-http"]


class MCPStdio(_MCPBase):
    transport: Literal["stdio"]
    command: str | list[str]
    args: list[str] = Field(default_factory=list)

    def argv(self) -> list[str]:
        base = (
            shlex.split(self.command)
            if isinstance(self.command, str)
            else list(self.command or [])
        )
        return [*base, *self.args] if self.args else base


MCPServer = Annotated[
    MCPHttp | MCPStreamableHttp | MCPStdio, Field(discriminator="transport")
]


class ModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    temperature: float = 0.2
    input_price: float = 0.0  # Price per million input tokens
    output_price: float = 0.0  # Price per million output tokens
    max_seq_length: int | None = None
    max_new_tokens: int | None = None
    load_in_4bit: bool | None = None
    dtype: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_alias_to_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "alias" not in data or data["alias"] is None:
                data["alias"] = data.get("name")
        return data


DEFAULT_PROVIDERS = [
    ProviderConfig(
        name="mistral",
        api_base="https://api.mistral.ai/v1",
        api_key_env_var="MISTRAL_API_KEY",
        backend=Backend.MISTRAL,
    ),
    ProviderConfig(
        name="llamacpp",
        api_base="http://127.0.0.1:8080/v1",
        api_key_env_var="",  # NOTE: if you wish to use --api-key in llama-server, change this value
    ),
]

DEFAULT_MODELS = [
    ModelConfig(
        name="mistral-vibe-cli-latest",
        provider="mistral",
        alias="devstral-2",
        input_price=0.4,
        output_price=2.0,
    ),
    ModelConfig(
        name="devstral-small-latest",
        provider="mistral",
        alias="devstral-small",
        input_price=0.1,
        output_price=0.3,
    ),
    ModelConfig(
        name="devstral",
        provider="llamacpp",
        alias="local",
        input_price=0.0,
        output_price=0.0,
    ),
]


class VibeConfig(BaseSettings):
    active_model: str = "devstral-2"
    vim_keybindings: bool = False
    disable_welcome_banner_animation: bool = False
    displayed_workdir: str = ""
    auto_compact_threshold: int = 200_000
    chat_history_max_entries: int = 100
    response_length: ResponseLength = ResponseLength.BALANCED
    response_tone: ResponseTone = ResponseTone.NEUTRAL
    response_task_style: ResponseTaskStyle = ResponseTaskStyle.STANDARD
    reasoning_mode: ReasoningMode = ReasoningMode.MEDIUM
    style_prompt: str = ""
    model_style_prompts: dict[str, str] = Field(default_factory=dict)
    context_warnings: bool = False
    textual_theme: str = "textual-dark"
    instructions: str = ""
    workdir: Path | None = Field(default=None, exclude=True)
    system_prompt_id: str = "cli"
    include_commit_signature: bool = True
    include_model_info: bool = True
    include_project_context: bool = True
    include_prompt_detail: bool = True
    enable_update_checks: bool = True
    api_timeout: float = 720.0
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_PROVIDERS)
    )
    models: list[ModelConfig] = Field(default_factory=lambda: list(DEFAULT_MODELS))

    project_context: ProjectContextConfig = Field(default_factory=ProjectContextConfig)
    project_mode: ProjectModeConfig = Field(default_factory=ProjectModeConfig)
    mistral_intelligence: MistralIntelligenceConfig = Field(
        default_factory=MistralIntelligenceConfig
    )
    extended_thinking: ExtendedThinkingConfig = Field(
        default_factory=ExtendedThinkingConfig
    )
    thought_mode: ThoughtModeConfig = Field(default_factory=ThoughtModeConfig)
    chain_of_thought: ChainOfThoughtConfig = Field(
        default_factory=ChainOfThoughtConfig
    )
    latex_mode: LatexModeConfig = Field(default_factory=LatexModeConfig)
    visual_layout: VisualLayoutConfig = Field(
        default_factory=VisualLayoutConfig
    )
    agent_full_access: AgentFullAccessConfig = Field(
        default_factory=AgentFullAccessConfig
    )
    codebase: CodebaseConfig = Field(default_factory=CodebaseConfig)
    session_logging: SessionLoggingConfig = Field(default_factory=SessionLoggingConfig)
    tools: dict[str, BaseToolConfig] = Field(default_factory=dict)
    tool_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for custom tools. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )

    mcp_servers: list[MCPServer] = Field(
        default_factory=list, description="Preferred MCP server configuration entries."
    )

    enabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of tool names/patterns to enable. If set, only these"
            " tools will be active. Supports exact names, glob patterns (e.g.,"
            " 'serena_*'), and regex with 're:' prefix or regex-like patterns (e.g.,"
            " 're:^serena_.*' or 'serena.*')."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "A list of tool names/patterns to disable. Ignored if 'enabled_tools'"
            " is set. Supports exact names, glob patterns (e.g., 'bash*'), and"
            " regex with 're:' prefix or regex-like patterns."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="VIBE_", case_sensitive=False, extra="ignore"
    )

    @property
    def effective_workdir(self) -> Path:
        return self.workdir if self.workdir is not None else Path.cwd()

    @property
    def system_prompt(self) -> str:
        try:
            return SystemPrompt[self.system_prompt_id.upper()].read()
        except KeyError:
            pass

        custom_sp_path = (PROMPT_DIR.path / self.system_prompt_id).with_suffix(".md")
        if not custom_sp_path.is_file():
            raise MissingPromptFileError(self.system_prompt_id, str(PROMPT_DIR.path))
        return custom_sp_path.read_text()

    def get_active_model(self) -> ModelConfig:
        for model in self.models:
            if model.alias == self.active_model:
                return model
        raise ValueError(
            f"Active model '{self.active_model}' not found in configuration."
        )

    def get_provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        for provider in self.providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"Provider '{model.provider}' for model '{model.name}' not found in configuration."
        )

    def get_model_style_prompt(self) -> str:
        if not self.model_style_prompts:
            return ""
        if prompt := self.model_style_prompts.get(self.active_model):
            return prompt
        try:
            model_name = self.get_active_model().name
        except ValueError:
            return ""
        return self.model_style_prompts.get(model_name, "")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define the priority of settings sources.

        Note: dotenv_settings is intentionally excluded. API keys and other
        non-config environment variables are stored in .env but loaded manually
        into os.environ for use by providers. Only VIBE_* prefixed environment
        variables (via env_settings) and TOML config are used for Pydantic settings.
        """
        return (
            init_settings,
            env_settings,
            TomlFileSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _check_api_key(self) -> VibeConfig:
        try:
            active_model = self.get_active_model()
            provider = self.get_provider_for_model(active_model)
            api_key_env = provider.api_key_env_var
            if api_key_env and not os.getenv(api_key_env):
                raise MissingAPIKeyError(api_key_env, provider.name)
        except ValueError:
            pass
        return self

    @model_validator(mode="after")
    def _check_api_backend_compatibility(self) -> VibeConfig:
        try:
            active_model = self.get_active_model()
            provider = self.get_provider_for_model(active_model)
            if provider.backend == Backend.UNSLOTH:
                return self
            MISTRAL_API_BASES = [
                "https://codestral.mistral.ai",
                "https://api.mistral.ai",
            ]
            is_mistral_api = any(
                provider.api_base.startswith(api_base) for api_base in MISTRAL_API_BASES
            )
            if (is_mistral_api and provider.backend != Backend.MISTRAL) or (
                not is_mistral_api and provider.backend != Backend.GENERIC
            ):
                raise WrongBackendError(provider.backend, is_mistral_api)

        except ValueError:
            pass
        return self

    @field_validator("workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: Any) -> Path | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None

        if isinstance(v, str):
            v = Path(v).expanduser().resolve()
        elif isinstance(v, Path):
            v = v.expanduser().resolve()
        if not v.is_dir():
            raise ValueError(
                f"Tried to set {v} as working directory, path doesn't exist"
            )
        return v

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tool_configs(cls, v: Any) -> dict[str, BaseToolConfig]:
        if not isinstance(v, dict):
            return {}

        normalized: dict[str, BaseToolConfig] = {}
        for tool_name, tool_config in v.items():
            if isinstance(tool_config, BaseToolConfig):
                normalized[tool_name] = tool_config
            elif isinstance(tool_config, dict):
                normalized[tool_name] = BaseToolConfig.model_validate(tool_config)
            else:
                normalized[tool_name] = BaseToolConfig()

        return normalized

    @field_validator(
        "response_length",
        "response_tone",
        "response_task_style",
        "reasoning_mode",
        mode="before",
    )
    @classmethod
    def _normalize_style_enum(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _validate_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _check_system_prompt(self) -> VibeConfig:
        _ = self.system_prompt
        return self

    @classmethod
    def save_updates(cls, updates: dict[str, Any]) -> None:
        CONFIG_DIR.path.mkdir(parents=True, exist_ok=True)
        current_config = TomlFileSettingsSource(cls).toml_data

        def deep_merge(target: dict, source: dict) -> None:
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target.get(key), dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(target[key], value)
                elif (
                    key in target
                    and isinstance(target.get(key), list)
                    and isinstance(value, list)
                ):
                    if key in {"providers", "models"}:
                        target[key] = value
                    else:
                        target[key] = list(set(value + target[key]))
                else:
                    target[key] = value

        deep_merge(current_config, updates)
        cls.dump_config(
            to_jsonable_python(current_config, exclude_none=True, fallback=str)
        )

    @classmethod
    def dump_config(cls, config: dict[str, Any]) -> None:
        with CONFIG_FILE.path.open("wb") as f:
            tomli_w.dump(config, f)

    @classmethod
    def _get_agent_config(cls, agent: str | None) -> dict[str, Any] | None:
        if agent is None:
            return None

        agent_config_path = (AGENT_DIR.path / agent).with_suffix(".toml")
        try:
            return tomllib.load(agent_config_path.open("rb"))
        except FileNotFoundError:
            raise ValueError(
                f"Config '{agent}.toml' for agent not found in {AGENT_DIR.path}"
            )

    @classmethod
    def _migrate(cls) -> None:
        if not CONFIG_FILE.path.is_file():
            return

        try:
            with CONFIG_FILE.path.open("rb") as f:
                config = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return

        needs_save = False

        def merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> bool:
            updated = False
            for key, value in defaults.items():
                if key not in target or target[key] is None:
                    target[key] = value
                    updated = True
                    continue
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    if merge_missing(target[key], value):
                        updated = True
            return updated

        try:
            defaults = cls.create_default()
        except Exception:
            defaults = {}

        if isinstance(defaults, dict) and defaults:
            if merge_missing(config, defaults):
                needs_save = True

        if (
            "auto_compact_threshold" not in config
            or config["auto_compact_threshold"] == 100_000  # noqa: PLR2004
        ):
            config["auto_compact_threshold"] = 200_000
            needs_save = True

        if needs_save:
            cls.dump_config(config)

    @classmethod
    def load(cls, agent: str | None = None, **overrides: Any) -> VibeConfig:
        cls._migrate()
        agent_config = cls._get_agent_config(agent)
        init_data = {**(agent_config or {}), **overrides}
        return cls(**init_data)

    @classmethod
    def create_default(cls) -> dict[str, Any]:
        try:
            config = cls()
        except MissingAPIKeyError:
            config = cls.model_construct()

        config_dict = config.model_dump(mode="json", exclude_none=True)

        try:
            from vibe.core.tools.manager import ToolManager

            tool_defaults = ToolManager.discover_tool_defaults()
        except Exception:
            tool_defaults = {}

        if tool_defaults:
            config_dict["tools"] = tool_defaults

        return config_dict
