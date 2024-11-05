import logging
import warnings
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    DefaultDict,
    List,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    cast,
    get_type_hints,
    overload,
    runtime_checkable,
)

from autogen_core.base import BaseAgent, MessageContext, MessageSerializer, try_get_known_serializers_for_type
from autogen_core.base._type_helpers import AnyType, get_types
from autogen_core.base.exceptions import CantHandleException

logger = logging.getLogger("autogen_core")

AgentT = TypeVar("AgentT")
ReceivesT = TypeVar("ReceivesT")
ProducesT = TypeVar("ProducesT", covariant=True)


from opto import trace
import opto.trace.operators as ops
from dataclasses import dataclass
from autogen_core.components import RoutedAgent
from autogen_core.components._routed_agent import MessageHandler
from autogen_core.base._topic import TopicId
from autogen_core.base._cancellation_token import CancellationToken

@dataclass
class TraceMessage:
    content: trace.Node
    def __init__(self, message: Any):
        self.content = trace.node(message)

class TracedRoutedAgent(RoutedAgent):


    def __init__(self, description: str) -> None:

        super().__init__(description)
        self._input_node = None  # HACK  to record who are the inputs

    async def on_message(self, message: Any, ctx: MessageContext) -> Any | None:
        if not type(message) == TraceMessage:
            message = TraceMessage(message)
        # Unwrapp the message
        content_node = message.content  # trace.Node
        self._input_node = content_node  # save the input node
        message = content_node._data

        return await super().on_message(message, ctx)

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        cancellation_token: CancellationToken | None = None,
        # XXX Need description, Maybe there is a way to generate it automatically
        description = ''
    ) -> None:

        # This is where we create the operator
        @trace.bundle(description=description)
        def bundled_func(inputs):
            return message

        output_node = bundled_func(self._input_node)
        message = TraceMessage(output_node)

        await super().publish_message(message, topic_id, cancellation_token=cancellation_token)
