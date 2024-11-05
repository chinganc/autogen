import asyncio
from dataclasses import dataclass
from typing import Literal

import logging

from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.application.logging import TRACE_LOGGER_NAME
from autogen_core.base import MessageContext
from autogen_core.components import RoutedAgent, message_handler, DefaultTopicId, default_subscription

from trace_decorator import TracedRoutedAgent, TraceMessage


def fun(x):
    return output


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

    @message_handler
    async def handle_start(self, message: Start, ctx: MessageContext) -> None:
        print(f"{'-'*80}\n{self.id.type} started")
        await self.publish_message(Problem(x=1, y=2, op='add'), DefaultTopicId(), description='[handle_start]')

    @message_handler
    async def handle_answer(self, message: Solution, ctx: MessageContext) -> None:
        if message.result == 3:
            print(f"{'-'*80}\n{self.id.type} got the correct answer")

            # XXX for demo purpuse
            last_node = self._input_node
            digraph = last_node.backward(visualize=True)
            digraph.render("trace", format="png", cleanup=True)
            return
        else:
            print(f"{'-'*80}\n{self.id.type} got the wrong answer")
            await self.publish_message(Problem(x=1, y=2, op='add'), DefaultTopicId(), description='[handle_answer]')


@default_subscription
class SolutionAgent(TracedRoutedAgent):

    @message_handler
    async def handle_problem(self, message: Problem, ctx: MessageContext) -> None:
        print(f"{'-'*80}\n{self.id.type} received problem: {message.x} {message.op} {message.y}")
        if message.op == 'add':
            solution = Solution(result=message.x + message.y)
        elif message.op == 'sub':
            solution = Solution(result=message.x - message.y)
        elif message.op == 'mul':
            solution =  Solution(result=message.x * message.y)
        elif message.op == 'div':
            solution = Solution(result=message.x // message.y)
        else:
            raise NotImplementedError(f"Unknown operation: {message.op}")

        await self.publish_message(solution, DefaultTopicId(), description='[handle_problem]')


async def main() -> None:

    # logging.basicConfig(level=logging.WARNING)
    # logger = logging.getLogger(TRACE_LOGGER_NAME)
    # logger.setLevel(logging.INFO)

    runtime = SingleThreadedAgentRuntime()

    await ProblemAgent.register(runtime, type="problem_agent", factory=lambda: ProblemAgent("Problem Agent"))
    await SolutionAgent.register(runtime, type="solution_agent", factory=lambda: SolutionAgent("Solution Agent"))

    runtime.start()
    await runtime.publish_message(Start(), DefaultTopicId())

    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())