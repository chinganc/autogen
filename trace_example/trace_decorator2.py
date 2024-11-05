import asyncio
from dataclasses import dataclass
from typing import Literal, Any, get_type_hints
from autogen_core.base._type_helpers import AnyType, get_types

import logging

from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.application.logging import TRACE_LOGGER_NAME
from autogen_core.base import MessageContext
from autogen_core.components import RoutedAgent, message_handler, DefaultTopicId, default_subscription


from opto import trace
import inspect


# def magic(func):

#     type_hints = get_type_hints(func)
#     target_types = get_types(type_hints["message"]) # # HACK

#     @message_handler
#     async def publisher(self, message: target_types[0], ctx: MessageContext) -> None:

#         output =  await func(self, message, ctx)
#         if output is not None:
#             output, topic_id = output
#             await self.publish_message(output, topic_id)

#     return publisher


@dataclass
class TraceMessage:
    content: trace.Node
    def __init__(self, message: Any):
        self.content = trace.node(message)

class TracedRoutedAgent(RoutedAgent):

    async def on_message(self, message: Any, ctx: MessageContext) -> Any | None:
        """Handle a message by routing it to the appropriate message handler.
        Do not override this method in subclasses. Instead, add message handlers as methods decorated with
        either the :func:`event` or :func:`rpc` decorator."""
        # XXX We route the message based on the type of the message content.
        if not type(message) == TraceMessage:
            message = TraceMessage(message)
        key_type = type(message.content._data)  # type: ignore
        # XXX

        ## key_type: Type[Any] = type(message)  # type: ignore
        handlers = self._handlers.get(key_type)  # type: ignore
        if handlers is not None:
            # Iterate over all handlers for this matching message type.
            # Call the first handler whose router returns True and then return the result.
            for h in handlers:
                if h.router(message, ctx):
                    return await h(self, message, ctx)
        return await self.on_unhandled_message(message, ctx)  # type: ignore

def magic(func):
    @message_handler
    async def publisher(self, message: TraceMessage, ctx: MessageContext) -> None:
        message = message.content  # trace.Node
        # Get info to define the operation
        func_name = func.__qualname__
        docstring = inspect.getdoc(func)
        doc = inspect.cleandoc(docstring) if docstring is not None else ""

        @trace.bundle(description=f'[{func_name}] {doc}')
        async def traced_func(message):  # so we only trace message
            return await func(self, message, ctx)

        output = traced_func(message)  # trace.MessageNode
         # XXX Do the actual computation
        output._data = await output._data  # TODO error not caught here; should modify bundle

        # HACK Remove the topic_id from the output
        topic_id = output[1].data
        output._data = output._data[0]

        if output._data is not None:
            await self.publish_message(TraceMessage(output), topic_id)
        else:  # XXX for demo purpuse only
            last_node = output
            digraph = last_node.backward(visualize=True)
            digraph.render("trace", format="png", cleanup=True)

    # HACK
    type_hints = get_type_hints(func)
    target_types = get_types(type_hints["message"]) #
    publisher.target_types = target_types  #  save the original target types
    return publisher