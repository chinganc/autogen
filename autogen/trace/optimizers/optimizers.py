from typing import Any, List, Dict, Union
from autogen.trace.nodes import ParameterNode, Node
from collections import defaultdict
from autogen import AssistantAgent
from autogen.oai.completion import Completion
from textwrap import dedent, indent
from copy import copy
from autogen.trace.propagators.propagators import Propagator
from dataclasses import dataclass
import autogen
import warnings
import json


class AbstractOptimizer:
    """An optimizer is responsible for updating the parameters based on the feedback."""

    def __init__(self, parameters: List[ParameterNode], *args, **kwargs):
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        self.parameters = parameters

    def step(self):
        """Update the parameters based on the feedback."""
        raise NotImplementedError

    def zero_feedback(self):
        """Reset the feedback."""
        raise NotImplementedError

    @property
    def propagator(self):
        """Return a Propagator object that can be used to propagate feedback in backward."""
        raise NotImplementedError


class Optimizer(AbstractOptimizer):
    def __init__(self, parameters: List[ParameterNode], *args, propagator: Propagator = None, **kwargs):
        super().__init__(parameters)
        propagator = propagator if propagator is not None else self.default_propagator()
        assert isinstance(propagator, Propagator)
        self._propagator = propagator

    @property
    def propagator(self):
        return self._propagator

    def step(self, *args, **kwargs):
        update_dict = self.propose(*args, **kwargs)
        self.update(update_dict)

    def propose(self, *args, **kwargs):
        """Propose the new data of the parameters based on the feedback."""
        return self._step(*args, **kwargs)

    def update(self, update_dict: Dict[ParameterNode, Any]):
        """Update the trainable parameters given a dictionary of new data."""
        for p, d in update_dict.items():
            if p.trainable:
                p._data = d

    def zero_feedback(self):
        for p in self.parameters:
            p.zero_feedback()

    # Subclass should implement the methods below.
    def _step(self, *args, **kwargs) -> Dict[ParameterNode, Any]:
        """Return the new data of parameter nodes based on the feedback."""
        raise NotImplementedError

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        raise NotImplementedError

    def backward(self, node: Node, *args, **kwargs):
        """Propagate the feedback backward."""
        return node.backward(*args, propagator=self.propagator, **kwargs)
