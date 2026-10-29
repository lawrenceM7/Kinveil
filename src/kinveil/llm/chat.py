from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING, Callable

from .base import ChatMessage, LanguageModel
from .context import BoundedContextBuilder

if TYPE_CHECKING:
    from kinveil.audio.voice import UserTurn
    from kinveil.memory import MemoryService


DEFAULT_SYSTEM_PROMPT = """你是 Kinveil，一个运行在用户设备上的私人 AI 助手。
请自然、简洁地回答当前说话者。
不要声称记得未出现在当前上下文中的信息。
记忆上下文只可用于当前已确认的用户，不要向其他人泄露私人信息。"""


class ChatSession:
    """Combines final attributed turns, bounded memories, and recent chat history."""

    def __init__(
        self,
        model: LanguageModel,
        context: BoundedContextBuilder,
        memories: MemoryService,
        *,
        maximum_history_messages: int = 20,
    ) -> None:
        self.model = model
        self.context = context
        self.memories = memories
        self.maximum_history_messages = maximum_history_messages
        self._history: dict[str, list[ChatMessage]] = defaultdict(list)
        self._lock = Lock()

    def respond(self, turn: UserTurn) -> str:
        confirmed = turn.person_id is not None and turn.identity_status == "confirmed"
        session_key = turn.person_id if confirmed else None
        memory_context = None
        if confirmed:
            recalled = self.memories.context_for(turn.person_id, turn.text)
            memory_context = recalled.text or None
        with self._lock:
            history = tuple(self._history[session_key]) if session_key is not None else ()
        bounded = self.context.build(
            system_messages=(ChatMessage("system", DEFAULT_SYSTEM_PROMPT),),
            history=history,
            current_user=ChatMessage("user", turn.text),
            memory_context=memory_context,
        )
        response = self.model.complete(bounded.request)
        if not response.text and response.tool_calls:
            return ""
        if session_key is not None:
            with self._lock:
                conversation = self._history[session_key]
                conversation.extend(
                    (ChatMessage("user", turn.text), ChatMessage("assistant", response.text))
                )
                del conversation[:-self.maximum_history_messages]
        return response.text


class BackgroundChatRunner:
    """Serializes model calls off the real-time audio/video loop."""

    def __init__(self, session: ChatSession, on_response: Callable[[UserTurn, str], None]) -> None:
        self.session = session
        self.on_response = on_response
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kinveil-chat")

    def submit(self, turn: UserTurn) -> Future[str]:
        future = self._executor.submit(self.session.respond, turn)

        def completed(result: Future[str]) -> None:
            try:
                text = result.result()
            except Exception as exc:
                self.on_response(turn, f"[chat error] {exc}")
                return
            self.on_response(turn, text)

        future.add_done_callback(completed)
        return future

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> BackgroundChatRunner:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
