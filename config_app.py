from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypedDict

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.theme import BUILTIN_THEMES
from textual.widgets import Static

from vibe.core.config import (
    ReasoningMode,
    ResponseLength,
    ResponseTaskStyle,
    ResponseTone,
)

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

THEMES = sorted(k for k in BUILTIN_THEMES if k != "textual-ansi")


def _bool_to_option(value: bool) -> str:
    return "on" if value else "off"


class SettingDefinition(TypedDict):
    key: str
    label: str
    type: str
    options: list[str]
    value: str


class ConfigApp(Container):
    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_setting", "Toggle", show=False),
        Binding("enter", "cycle", "Next", show=False),
    ]

    class SettingChanged(Message):
        def __init__(self, key: str, value: str) -> None:
            super().__init__()
            self.key = key
            self.value = value

    class ConfigClosed(Message):
        def __init__(self, changes: dict[str, str]) -> None:
            super().__init__()
            self.changes = changes

    def __init__(self, config: VibeConfig) -> None:
        super().__init__(id="config-app")
        self.config = config
        self.selected_index = 0
        self.changes: dict[str, str] = {}
        bool_options = ["off", "on"]

        self.settings: list[SettingDefinition] = [
            {
                "key": "active_model",
                "label": "Model",
                "type": "cycle",
                "options": [m.alias for m in self.config.models],
                "value": self.config.active_model,
            },
            {
                "key": "textual_theme",
                "label": "Theme",
                "type": "cycle",
                "options": THEMES,
                "value": self.config.textual_theme,
            },
            {
                "key": "response_length",
                "label": "Response length",
                "type": "cycle",
                "options": [option.value for option in ResponseLength],
                "value": self.config.response_length.value,
            },
            {
                "key": "response_tone",
                "label": "Response tone",
                "type": "cycle",
                "options": [option.value for option in ResponseTone],
                "value": self.config.response_tone.value,
            },
            {
                "key": "response_task_style",
                "label": "Task style",
                "type": "cycle",
                "options": [option.value for option in ResponseTaskStyle],
                "value": self.config.response_task_style.value,
            },
            {
                "key": "reasoning_mode",
                "label": "Reasoning mode",
                "type": "cycle",
                "options": [option.value for option in ReasoningMode],
                "value": self.config.reasoning_mode.value,
            },
            {
                "key": "project_mode_enabled",
                "label": "Project mode",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.project_mode.enabled),
            },
            {
                "key": "codebase_enabled",
                "label": "Codebase mode",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.codebase.enabled),
            },
            {
                "key": "mistral_intelligence_enabled",
                "label": "Mistral intelligence",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.mistral_intelligence.enabled),
            },
            {
                "key": "extended_thinking_enabled",
                "label": "Extended thinking",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.extended_thinking.enabled),
            },
            {
                "key": "thought_mode_enabled",
                "label": "Thought mode",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.thought_mode.enabled),
            },
            {
                "key": "chain_of_thought_enabled",
                "label": "Chain-of-thought",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.chain_of_thought.enabled),
            },
            {
                "key": "latex_mode_enabled",
                "label": "LaTeX mode",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.latex_mode.enabled),
            },
            {
                "key": "visual_layout_enabled",
                "label": "Visual layout",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.visual_layout.enabled),
            },
            {
                "key": "agent_full_access_enabled",
                "label": "Agent full access",
                "type": "cycle",
                "options": bool_options,
                "value": _bool_to_option(self.config.agent_full_access.enabled),
            },
        ]

        self.title_widget: Static | None = None
        self.setting_widgets: list[Static] = []
        self.help_widget: Static | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="config-content"):
            self.title_widget = Static("Settings", classes="settings-title")
            yield self.title_widget

            yield Static("")

            for _ in self.settings:
                widget = Static("", classes="settings-option")
                self.setting_widgets.append(widget)
                yield widget

            yield Static("")

            self.help_widget = Static(
                "↑↓ navigate  Space/Enter toggle  ESC exit", classes="settings-help"
            )
            yield self.help_widget

    def on_mount(self) -> None:
        self._update_display()
        self.focus()

    def _update_display(self) -> None:
        for i, (setting, widget) in enumerate(
            zip(self.settings, self.setting_widgets, strict=True)
        ):
            is_selected = i == self.selected_index
            cursor = "› " if is_selected else "  "

            label: str = setting["label"]
            value: str = self.changes.get(setting["key"], setting["value"])

            text = f"{cursor}{label}: {value}"

            widget.update(text)

            widget.remove_class("settings-cursor-selected")
            widget.remove_class("settings-value-cycle-selected")
            widget.remove_class("settings-value-cycle-unselected")

            if is_selected:
                widget.add_class("settings-value-cycle-selected")
            else:
                widget.add_class("settings-value-cycle-unselected")

    def action_move_up(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(self.settings)
        self._update_display()

    def action_move_down(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(self.settings)
        self._update_display()

    def action_toggle_setting(self) -> None:
        setting = self.settings[self.selected_index]
        key: str = setting["key"]
        current: str = self.changes.get(key, setting["value"])

        options: list[str] = setting["options"]
        try:
            current_idx = options.index(current)
            next_idx = (current_idx + 1) % len(options)
            new_value: str = options[next_idx]
        except (ValueError, IndexError):
            new_value: str = options[0] if options else current

        self.changes[key] = new_value

        self.post_message(self.SettingChanged(key=key, value=new_value))

        self._update_display()

    def action_cycle(self) -> None:
        self.action_toggle_setting()

    def action_close(self) -> None:
        self.post_message(self.ConfigClosed(changes=self.changes.copy()))

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
