from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Mapping, Protocol, TypeVar

MsgT = TypeVar("MsgT")
CtxT = TypeVar("CtxT")
StateName = str


class AsyncState(Protocol[CtxT, MsgT]):
    name: StateName

    async def on_enter(self, ctx: CtxT) -> None: ...

    async def on_exit(self, ctx: CtxT) -> None: ...

    async def handle(self, ctx: CtxT, msg: MsgT) -> StateName | None: ...


@dataclass(slots=True)
class AsyncStateMachine(Generic[CtxT, MsgT]):
    """
    Tiny async state machine.

    - States decide transitions by returning the next state's `name`.
    - If a state returns None, the machine stays in the current state.
    """

    states: Mapping[StateName, AsyncState[CtxT, MsgT]]
    initial: StateName
    _current: StateName | None = None

    @property
    def current(self) -> StateName:
        if self._current is None:
            return self.initial
        return self._current

    async def start(self, ctx: CtxT) -> None:
        if self._current is not None:
            return
        self._current = self.initial
        await self.states[self._current].on_enter(ctx)

    async def dispatch(self, ctx: CtxT, msg: MsgT) -> None:
        if self._current is None:
            await self.start(ctx)

        state = self.states[self._current]
        next_name = await state.handle(ctx, msg)
        if next_name is None or next_name == self._current:
            return

        await state.on_exit(ctx)
        self._current = next_name
        await self.states[self._current].on_enter(ctx)

