import asyncio
from dataclasses import dataclass
from typing import Literal

import logging

from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.application.logging import TRACE_LOGGER_NAME
from autogen_core.base import MessageContext
from autogen_core.components import RoutedAgent, message_handler, DefaultTopicId, default_subscription

from trace_decorator2 import magic, TraceMessage, TracedRoutedAgent

@dataclass
class Start:
    pass


@dataclass
class Problem:
    x: int
    y: int
    op: Literal['add', 'sub', 'mul', 'div']


@dataclass
class Solution:
    result: int

@dataclass
class Feedback:
    correct: bool


@default_subscription
class ProblemAgent(TracedRoutedAgent):

    @magic
    async def handle_start(self, message: Start, ctx: MessageContext):
        """ Handle a start message """
        print(f"{'-'*80}\n{self.id.type} started")
        return Problem(x=1, y=2, op='add'), DefaultTopicId()
        # await self.publish_message(Problem(x=1, y=2, op='add'), DefaultTopicId())

    @magic
    async def handle_answer(self, message: Solution, ctx: MessageContext):
        """ Handle an answer message """
        if message.result == 3:
            print(f"{'-'*80}\n{self.id.type} got the correct answer")
            return None, None
        else:
            print(f"{'-'*80}\n{self.id.type} got the wrong answer")
            return Problem(x=1, y=2, op='add'), DefaultTopicId()

@default_subscription
class SolutionAgent(TracedRoutedAgent):

    @magic
    async def handle_problem(self, message: Problem, ctx: MessageContext) -> None:
        """ Handle a problem message """
        print(f"{'-'*80}\n{self.id.type} received problem: {message.x} {message.op} {message.y}")
        if message.op == 'add':
            # await self.publish_message(Solution(result=message.x + message.y), DefaultTopicId())
            return Solution(result=message.x + message.y), DefaultTopicId()
        elif message.op == 'sub':
            # await self.publish_message(Solution(result=message.x - message.y), DefaultTopicId())
            return Solution(result=message.x - message.y), DefaultTopicId()
        elif message.op == 'mul':
            # await self.publish_message(Solution(result=message.x * message.y), DefaultTopicId())
            return Solution(result=message.x * message.y), DefaultTopicId()
        elif message.op == 'div':
            # await self.publish_message(Solution(result=message.x // message.y), DefaultTopicId())
            return Solution(result=message.x // message.y), DefaultTopicId()
        else:
            raise NotImplementedError(f"Unknown operation: {message.op}")


async def main() -> None:

    # logging.basicConfig(level=logging.WARNING)
    # logger = logging.getLogger(TRACE_LOGGER_NAME)
    # logger.setLevel(logging.INFO)

    runtime = SingleThreadedAgentRuntime()

    await ProblemAgent.register(runtime, type="problem_agent", factory=lambda: ProblemAgent("Problem Agent"))
    await SolutionAgent.register(runtime, type="solution_agent", factory=lambda: SolutionAgent("Solution Agent"))

    runtime.start()
    # await runtime.publish_message(Start(), DefaultTopicId())
    await runtime.publish_message(TraceMessage(Start()), DefaultTopicId())

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())