import sys

sys.path.append(".")
sys.path.append("C:/Users/adswamin/NegotiationArena/")

from dotenv import load_dotenv
import inspect
from ratbench.agents import *
from ratbench.agents.agent_behaviours import SelfCheckingAgent, ReasoningAgent
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import ResourceGoal
from games.trading_game.game import TradingGame
from games.trading_game.interface import TradingGameInterface
from ratbench.constants import *

load_dotenv(".env")

import autogen
from autogen.trace.nodes import ParameterNode

if __name__ == "__main__":
    for i in range(1):
        a1 = ChatGPTAgent(
            model="gpt-4-1106-preview",
            agent_name=AGENT_ONE,
        )
        a2 = ChatGPTAgent(
            model="gpt-4-1106-preview",
            agent_name=AGENT_TWO,
        )

        a1_social_prompt = ParameterNode("You want a win-win solution.", trainable=True)
        a2_social_prompt = ParameterNode("You want a win-win solution.", trainable=True)
                                      
        c = TradingGame(
            players=[a1, a2],
            iterations=6,
            resources_support_set=Resources({"X": 0, "Y": 0}),
            player_goals=[
                ResourceGoal({"X": 15, "Y": 15}),
                ResourceGoal({"X": 15, "Y": 15}),
            ],
            player_initial_resources=[
                Resources({"X": 25, "Y": 5}),
                Resources({"X": 5, "Y": 25}),
            ],
            player_social_behaviour=[a1_social_prompt, a2_social_prompt],
            player_roles=[
                f"You are {AGENT_ONE}, start by making a proposal.",
                f"You are {AGENT_TWO}, start by responding to a trade.",
            ],
            log_dir="./.logs/trading/",
        )

        returned_node = c.run()
        print(returned_node)
        returned_node.backward(visualize=True)
