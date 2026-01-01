from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import types
from typing import TYPE_CHECKING, Any

from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig
    from vibe.core.types import AvailableTool, StrToolChoice


class UnslothBackend:
    _model_cache: dict[str, tuple[Any, Any]] = {}
    _model_lock = asyncio.Lock()

    def __init__(self, provider: ProviderConfig, timeout: float = 720.0) -> None:
        self._provider = provider
        self._timeout = timeout

    async def __aenter__(self) -> UnslothBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> LLMChunk:
        model_obj, tokenizer = await self._get_model(model)
        prompt = self._build_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        device = self._infer_device(model_obj)
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        max_new_tokens = self._resolve_max_new_tokens(model, max_tokens)
        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        outputs = await asyncio.to_thread(model_obj.generate, **inputs, **gen_kwargs)
        output_ids = outputs[0]
        if input_ids is not None:
            output_ids = output_ids[input_ids.shape[-1] :]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        prompt_tokens = int(input_ids.shape[-1]) if input_ids is not None else 0
        completion_tokens = int(output_ids.shape[-1]) if output_ids is not None else 0

        return LLMChunk(
            message=LLMMessage(role=Role.assistant, content=text),
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason="stop",
        )

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> AsyncGenerator[LLMChunk, None]:
        chunk = await self.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            extra_headers=extra_headers,
        )
        yield chunk

    async def count_tokens(
        self,
        *,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None,
    ) -> int:
        _, tokenizer = await self._get_model(model)
        prompt = self._build_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        return int(input_ids.shape[-1]) if input_ids is not None else 0

    async def _get_model(self, model: ModelConfig) -> tuple[Any, Any]:
        cache_key = self._cache_key(model)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        async with self._model_lock:
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

            model_obj, tokenizer = await asyncio.to_thread(
                self._load_model_sync, model
            )
            self._model_cache[cache_key] = (model_obj, tokenizer)
            return model_obj, tokenizer

    def _load_model_sync(self, model: ModelConfig) -> tuple[Any, Any]:
        try:
            from unsloth import FastLanguageModel
        except Exception as exc:
            raise RuntimeError(
                "Unsloth is not installed. Install it with `pip install unsloth`."
            ) from exc

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for Unsloth. Install it with `pip install torch`."
            ) from exc

        dtype = self._resolve_dtype(model.dtype, torch)
        max_seq_length = model.max_seq_length or 2048
        load_in_4bit = model.load_in_4bit if model.load_in_4bit is not None else True

        model_obj, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model.name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model_obj)
        model_obj.eval()
        return model_obj, tokenizer

    def _resolve_dtype(self, value: str | None, torch: Any) -> Any | None:
        if not value or value.strip().lower() in {"auto", "none"}:
            return None
        normalized = value.strip().lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(normalized, None)

    def _resolve_max_new_tokens(
        self, model: ModelConfig, max_tokens: int | None
    ) -> int:
        fallback = model.max_new_tokens or 512
        if max_tokens is None:
            return fallback
        if model.max_new_tokens is None:
            return max_tokens
        return min(max_tokens, model.max_new_tokens)

    def _cache_key(self, model: ModelConfig) -> str:
        return (
            f"{model.name}|"
            f"{model.max_seq_length}|"
            f"{model.load_in_4bit}|"
            f"{model.dtype}"
        )

    def _build_prompt(self, messages: list[LLMMessage], tokenizer: Any) -> str:
        chat_messages = []
        for msg in messages:
            content = (msg.content or "").strip()
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if msg.role == Role.tool:
                role = "user"
                tool_name = msg.name or "tool"
                content = f"[tool:{tool_name}] {content}"
            chat_messages.append({"role": role, "content": content})

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        parts = []
        for msg in chat_messages:
            role = msg["role"].upper()
            content = msg["content"]
            parts.append(f"[{role}]\n{content}")
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts).strip()

    def _infer_device(self, model_obj: Any) -> Any:
        try:
            import torch
        except Exception:
            return None
        try:
            return next(model_obj.parameters()).device
        except Exception:
            return torch.device("cpu")
