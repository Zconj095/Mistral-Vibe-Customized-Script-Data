from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    aliases: frozenset[str]
    description: str
    handler: str
    exits: bool = False


class CommandRegistry:
    def __init__(self, excluded_commands: list[str] | None = None) -> None:
        if excluded_commands is None:
            excluded_commands = []
        self.commands = {
            "help": Command(
                aliases=frozenset(["/help", "/h"]),
                description="Show help message",
                handler="_show_help",
            ),
            "status": Command(
                aliases=frozenset(["/status", "/stats"]),
                description="Display agent statistics",
                handler="_show_status",
            ),
            "config": Command(
                aliases=frozenset(["/config", "/cfg", "/theme", "/model"]),
                description="Edit config settings",
                handler="_show_config",
            ),
            "codebase": Command(
                aliases=frozenset(["/codebase"]),
                description="Toggle codebase mode on/off",
                handler="_toggle_codebase_mode",
            ),
            "codebase_index": Command(
                aliases=frozenset(["/codebase-index", "/codebase-scan"]),
                description="Build or refresh the codebase index",
                handler="_codebase_index",
            ),
            "codebase_summary": Command(
                aliases=frozenset(["/codebase-summary"]),
                description="Generate a codebase summary",
                handler="_codebase_summary",
            ),
            "codebase_graph": Command(
                aliases=frozenset(["/codebase-graph"]),
                description="Show a dependency graph summary",
                handler="_codebase_graph",
            ),
            "codebase_status": Command(
                aliases=frozenset(["/codebase-status"]),
                description="Show codebase mode status",
                handler="_codebase_status",
            ),
            "project": Command(
                aliases=frozenset(["/project", "/project-mode"]),
                description="Toggle project mode on/off",
                handler="_toggle_project_mode",
            ),
            "project_status": Command(
                aliases=frozenset(["/project-status"]),
                description="Show project mode status",
                handler="_project_status",
            ),
            "mistral_intelligence": Command(
                aliases=frozenset(
                    ["/mistral-intelligence", "/mi", "/mistral-intel"]
                ),
                description="Toggle Mistral intelligence mode on/off",
                handler="_toggle_mistral_intelligence",
            ),
            "mistral_intelligence_status": Command(
                aliases=frozenset(
                    ["/mistral-intelligence-status", "/mi-status"]
                ),
                description="Show Mistral intelligence mode status",
                handler="_mistral_intelligence_status",
            ),
            "extended_thinking": Command(
                aliases=frozenset(
                    ["/extended-thinking", "/extended", "/think"]
                ),
                description="Toggle extended thinking mode on/off",
                handler="_toggle_extended_thinking",
            ),
            "extended_thinking_status": Command(
                aliases=frozenset(
                    ["/extended-thinking-status", "/think-status"]
                ),
                description="Show extended thinking mode status",
                handler="_extended_thinking_status",
            ),
            "thought_mode": Command(
                aliases=frozenset(["/thought-mode", "/thought"]),
                description="Toggle thought mode on/off",
                handler="_toggle_thought_mode",
            ),
            "thought_mode_status": Command(
                aliases=frozenset(["/thought-status"]),
                description="Show thought mode status",
                handler="_thought_mode_status",
            ),
            "chain_of_thought": Command(
                aliases=frozenset(
                    ["/chain-of-thought", "/cot", "/cot-mode"]
                ),
                description="Toggle chain-of-thought mode on/off",
                handler="_toggle_chain_of_thought",
            ),
            "chain_of_thought_status": Command(
                aliases=frozenset(
                    ["/chain-of-thought-status", "/cot-status"]
                ),
                description="Show chain-of-thought mode status",
                handler="_chain_of_thought_status",
            ),
            "reasoning_mode": Command(
                aliases=frozenset(["/reasoning-mode", "/reasoning"]),
                description="Cycle reasoning mode",
                handler="_cycle_reasoning_mode",
            ),
            "reasoning_low": Command(
                aliases=frozenset(["/reasoning-low"]),
                description="Set reasoning mode to low",
                handler="_set_reasoning_low",
            ),
            "reasoning_medium": Command(
                aliases=frozenset(["/reasoning-medium"]),
                description="Set reasoning mode to medium",
                handler="_set_reasoning_medium",
            ),
            "reasoning_high": Command(
                aliases=frozenset(["/reasoning-high"]),
                description="Set reasoning mode to high",
                handler="_set_reasoning_high",
            ),
            "reasoning_extra_high": Command(
                aliases=frozenset(["/reasoning-extra-high"]),
                description="Set reasoning mode to extra high",
                handler="_set_reasoning_extra_high",
            ),
            "reasoning_status": Command(
                aliases=frozenset(["/reasoning-status"]),
                description="Show reasoning mode status",
                handler="_reasoning_status",
            ),
            "latex_mode": Command(
                aliases=frozenset(["/latex", "/latex-mode"]),
                description="Toggle LaTeX mode on/off",
                handler="_toggle_latex_mode",
            ),
            "latex_mode_status": Command(
                aliases=frozenset(["/latex-status"]),
                description="Show LaTeX mode status",
                handler="_latex_mode_status",
            ),
            "visual_layout": Command(
                aliases=frozenset(
                    ["/visual-layout", "/layout-mode", "/layout"]
                ),
                description="Toggle visual layout mode on/off",
                handler="_toggle_visual_layout",
            ),
            "visual_layout_status": Command(
                aliases=frozenset(
                    ["/visual-layout-status", "/layout-status"]
                ),
                description="Show visual layout mode status",
                handler="_visual_layout_status",
            ),
            "agent_full_access": Command(
                aliases=frozenset(["/agent-full-access", "/full-access"]),
                description="Toggle agent full access on/off",
                handler="_toggle_agent_full_access",
            ),
            "agent_full_access_status": Command(
                aliases=frozenset(
                    ["/agent-full-access-status", "/full-access-status"]
                ),
                description="Show agent full access status",
                handler="_agent_full_access_status",
            ),
            "artifact": Command(
                aliases=frozenset(["/artifact", "/artifacts"]),
                description="Start a Claude-style artifact flow",
                handler="_start_artifact",
            ),
            "screen_share": Command(
                aliases=frozenset(["/screen-share", "/screenshot"]),
                description="Capture a screenshot and OCR it",
                handler="_start_screen_share",
            ),
            "timeline_recall": Command(
                aliases=frozenset(
                    ["/timeline", "/timeline-recall", "/recall"]
                ),
                description="Recall events from sessions or text sources",
                handler="_start_timeline_recall",
            ),
            "timeline_code": Command(
                aliases=frozenset(
                    ["/timeline-code", "/code-timeline", "/code-recall"]
                ),
                description="Recall a timeline of generated code",
                handler="_start_timeline_code",
            ),
            "timeline_reason": Command(
                aliases=frozenset(
                    ["/timeline-reason", "/timeline-analyze", "/timeline-insight"]
                ),
                description="Reason across multiple timelines",
                handler="_start_timeline_reason",
            ),
            "document_code_synthesis": Command(
                aliases=frozenset(
                    ["/doc-code", "/doc-synth", "/doc-build", "/doc-spec"]
                ),
                description="Extract requirements from documents for code generation",
                handler="_start_document_code_synthesis",
            ),
            "latex_equation_synthesis": Command(
                aliases=frozenset(
                    ["/latex-equations", "/latex-synth", "/latex-spec"]
                ),
                description="Extract LaTeX equations for code generation",
                handler="_start_latex_equation_synthesis",
            ),
            "natural_language_synthesis": Command(
                aliases=frozenset(
                    ["/text-synth", "/nl-synth", "/synthesize-text"]
                ),
                description="Synthesize long-form text documents",
                handler="_start_natural_language_synthesis",
            ),
            "text_syntax_alignment": Command(
                aliases=frozenset(
                    ["/text-align", "/syntax-align", "/align-text"]
                ),
                description="Align syntactically related sentences in text",
                handler="_start_text_syntax_alignment",
            ),
            "text_syntax_grammar_multi": Command(
                aliases=frozenset(
                    [
                        "/text-grammar",
                        "/syntax-grammar",
                        "/grammar-analyze",
                    ]
                ),
                description="Analyze syntax and grammar across documents",
                handler="_start_text_syntax_grammar_multi",
            ),
            "text_semantic_correlation": Command(
                aliases=frozenset(
                    ["/text-correlate", "/semantic-correlate", "/text-semantic"]
                ),
                description="Correlate semantically similar segments in a document",
                handler="_start_text_semantic_correlation",
            ),
            "text_semantic_correlation_multi": Command(
                aliases=frozenset(
                    [
                        "/text-correlate-multi",
                        "/semantic-correlate-multi",
                        "/text-semantic-multi",
                    ]
                ),
                description="Correlate semantics across multiple documents",
                handler="_start_text_semantic_correlation_multi",
            ),
            "text_dense_layer_search": Command(
                aliases=frozenset(
                    ["/dense-layers", "/dense-search", "/context-search"]
                ),
                description="Search dense context layers in a document",
                handler="_start_text_dense_layer_search",
            ),
            "text_recursive_dense_search": Command(
                aliases=frozenset(
                    ["/dense-recursion", "/recursive-dense", "/dense-sectors"]
                ),
                description="Search dense recursion layers and sectors in a document",
                handler="_start_text_recursive_dense_search",
            ),
            "text_permutation_architecture": Command(
                aliases=frozenset(
                    ["/text-permute", "/permute-text", "/permute-arch"]
                ),
                description="Generate permutative linguistic architectures",
                handler="_start_text_permutation_architecture",
            ),
            "text_permutation_architecture_multi": Command(
                aliases=frozenset(
                    ["/text-permute-multi", "/permute-multi", "/permute-docs"]
                ),
                description="Generate permutative architectures across documents",
                handler="_start_text_permutation_architecture_multi",
            ),
            "text_subprocess_context_multi": Command(
                aliases=frozenset(
                    ["/text-subprocess", "/context-subprocess", "/multi-context"]
                ),
                description="Subprocess text regions with multi-context links",
                handler="_start_text_subprocess_context_multi",
            ),
            "text_context_multishape_multi": Command(
                aliases=frozenset(
                    [
                        "/text-context-shapes",
                        "/text-context-multi",
                        "/context-shapes",
                    ]
                ),
                description="Correlate multi-shape contexts across documents",
                handler="_start_text_context_multishape_multi",
            ),
            "text_context_unitize_harmonize_multi": Command(
                aliases=frozenset(
                    [
                        "/context-harmonize",
                        "/unitize-context",
                        "/harmonize-context",
                    ]
                ),
                description="Unitize and harmonize contexts across documents",
                handler="_start_text_context_unitize_harmonize_multi",
            ),
            "text_context_recursive_crossref_multi": Command(
                aliases=frozenset(
                    [
                        "/recursive-crossref",
                        "/context-recursive",
                        "/context-crossref",
                    ]
                ),
                description="Cross-reference contexts with recursive scoring",
                handler="_start_text_context_recursive_crossref_multi",
            ),
            "text_chunk_stream_multi": Command(
                aliases=frozenset(
                    [
                        "/chunk-stream",
                        "/chunk-multi",
                        "/chunk-massive",
                    ]
                ),
                description="Stream chunks across many documents",
                handler="_start_text_chunk_stream_multi",
            ),
            "text_structure_retrieval_multi": Command(
                aliases=frozenset(
                    [
                        "/structure-search",
                        "/structure-retrieve",
                        "/multi-structure",
                    ]
                ),
                description="Retrieve documents using multi-structure scoring",
                handler="_start_text_structure_retrieval_multi",
            ),
            "text_word_serialize_multi": Command(
                aliases=frozenset(
                    [
                        "/word-serialize",
                        "/word-inventory",
                        "/word-list",
                    ]
                ),
                description="Serialize words across documents",
                handler="_start_text_word_serialize_multi",
            ),
            "text_word_registry_multi": Command(
                aliases=frozenset(
                    [
                        "/word-registry",
                        "/word-network",
                        "/word-graph",
                    ]
                ),
                description="Build interconnected word registries across documents",
                handler="_start_text_word_registry_multi",
            ),
            "text_domain_correlation_multi": Command(
                aliases=frozenset(
                    [
                        "/text-domain-correlate",
                        "/domain-correlate",
                        "/domain-contexts",
                    ]
                ),
                description="Correlate contexts across documents with domain scoring",
                handler="_start_text_domain_correlation_multi",
            ),
            "text_embedding_synapse_multi": Command(
                aliases=frozenset(
                    [
                        "/embedding-synapse",
                        "/synapse-embed",
                        "/transformer-correlate",
                    ]
                ),
                description="Correlate contexts across multiple embedding models",
                handler="_start_text_embedding_synapse_multi",
            ),
            "text_meta_fluency_multi": Command(
                aliases=frozenset(
                    ["/meta-fluency", "/text-fluency", "/meta-cognitive"]
                ),
                description="Analyze meta cognitive fluency across documents",
                handler="_start_text_meta_fluency_multi",
            ),
            "text_recursive_layers_multi": Command(
                aliases=frozenset(
                    ["/recursive-layers", "/text-recursive", "/layer-recursive"]
                ),
                description="Read text across recursive layers",
                handler="_start_text_recursive_layers_multi",
            ),
            "gptoss_model": Command(
                aliases=frozenset(["/gptoss-model", "/gptoss-new"]),
                description="Create a custom GPT-OSS model profile",
                handler="_start_gptoss_model",
            ),
            "gem_model": Command(
                aliases=frozenset(["/gem-model", "/gem-new"]),
                description="Create a custom GEM model profile",
                handler="_start_gem_model",
            ),
            "latex_process": Command(
                aliases=frozenset(["/latex-process"]),
                description="Process LaTeX with math extraction or PDF compile",
                handler="_start_latex_process",
            ),
            "reload": Command(
                aliases=frozenset(["/reload", "/r"]),
                description="Reload configuration from disk",
                handler="_reload_config",
            ),
            "clear": Command(
                aliases=frozenset(["/clear", "/reset"]),
                description="Clear conversation history",
                handler="_clear_history",
            ),
            "log": Command(
                aliases=frozenset(["/log", "/logpath"]),
                description="Show path to current interaction log file",
                handler="_show_log_path",
            ),
            "compact": Command(
                aliases=frozenset(["/compact", "/summarize"]),
                description="Compact conversation history by summarizing",
                handler="_compact_history",
            ),
            "exit": Command(
                aliases=frozenset(["/exit", "/quit", "/q"]),
                description="Exit the application",
                handler="_exit_app",
                exits=True,
            ),
        }

        for command in excluded_commands:
            self.commands.pop(command, None)

        self._alias_map = {}
        for cmd_name, cmd in self.commands.items():
            for alias in cmd.aliases:
                self._alias_map[alias] = cmd_name

    def find_command(self, user_input: str) -> Command | None:
        cmd_name = self._alias_map.get(user_input.lower().strip())
        return self.commands.get(cmd_name) if cmd_name else None

    def get_help_text(self) -> str:
        lines: list[str] = [
            "### Keyboard Shortcuts",
            "",
            "- `Enter` Submit message",
            "- `Ctrl+J` / `Shift+Enter` Insert newline",
            "- `Escape` Interrupt agent or close dialogs",
            "- `Ctrl+C` Quit (or clear input if text present)",
            "- `Ctrl+O` Toggle tool output view",
            "- `Ctrl+T` Toggle todo view",
            "- `Shift+Tab` Toggle auto-approve mode",
            "",
            "### Special Features",
            "",
            "- `!<command>` Execute bash command directly",
            "- `@path/to/file/` Autocompletes file paths",
            "",
            "### Commands",
            "",
        ]

        for cmd in self.commands.values():
            aliases = ", ".join(f"`{alias}`" for alias in sorted(cmd.aliases))
            lines.append(f"- {aliases}: {cmd.description}")
        return "\n".join(lines)
