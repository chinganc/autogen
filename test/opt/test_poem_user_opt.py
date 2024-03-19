"""
In this file, we should have:
2. Add FeedbackEnhance
3. An optimizer that optimizes...with an agent call (20 min)
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, Agent
from autogen.trace.trace import trace, compatibility, node, trace_class
from autogen.trace.optimizers import TeacherLLMOptimizer
from autogen.trace.optimizer_autogen import train_with_env
from autogen.trace.utils import backfill_lists, plot_agent_performance, verbalize
from textwrap import dedent, indent
import llfbench

from autogen.trace.optimizers import DummyOptimizer

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo"],
    },
)
assert len(config_list) > 0


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


sys_msg1 = dedent("You are a student and your teacher gives you an assignment to write a poem.")


class PoemStudentAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="PoemStudentAgent",
            system_message=sys_msg1,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
            code_execution_config=False,
        )


sys_msg2 = dedent(
    "You are extracting a poem from the student's message. " + "Do not extract anything except the poem itself."
    "If the student did not write a poem, return an empty string."
)


class PoemExtractor(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="PoemExtractor",
            system_message=sys_msg2,
            llm_config={"temperature": 0.0, "config_list": config_list},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
            code_execution_config=False,
        )


# We inherit from the traced version of AssistantAgent and register new reply_funcs that based on nodes.
# class PoemAgent(trace(AssistantAgent, wrap_all_replies=False)):
@trace_class
class PoemAgent(AssistantAgent):
    def __init__(self, seed=456, silent=False):
        super().__init__(
            name="PoemAgent",
            system_message="",
            llm_config={"temperature": 0.0, "config_list": config_list, "cache_seed": seed},
            max_consecutive_auto_reply=1,
            is_termination_msg=termination_msg,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.student_agent = trace(PoemStudentAgent)()
        self.extractor_agent = trace(PoemExtractor)()

        self.poem = None
        self.silent = silent

        self.register_reply(UserProxyAgent, PoemAgent._generate_poem_reply, position=5)
        self.register_reply([PoemStudentAgent], PoemAgent._reply_to_terminate_agent)
        self.register_reply([PoemExtractor], PoemAgent._reply_to_terminate_extractor)

        # self.stop_reply_at_receive(self.student_agent)
        # self.stop_reply_at_receive(self.extractor_agent)

    def get_last_user_message(self, agent):
        for m in reversed(self.chat_message_nodes[agent]):
            if m.data["role"] == "user":
                return m

    def _generate_poem_reply(self, messages=None, sender=None, config=None):
        # message = messages[-1]['content']
        message = messages[-1]

        if self.poem is None:
            self.initiate_chat(self.student_agent, message=message, clear_history=True, silent=self.silent)
            self.poem = self.get_last_user_message(self.student_agent)  # ["content"]

        # this just means we haven't called extractor agent before
        if len(self._oai_messages[self.extractor_agent]) == 0:
            self.initiate_chat(self.extractor_agent, message=self.poem, clear_history=True, silent=self.silent)

        # extracted_poem = self.get_last_user_message(self.extractor_agent)["content"]
        extracted_poem = self.get_last_user_message(self.extractor_agent)  # ["content"]

        return node(True), extracted_poem  # {"content": extracted_poem}

    def _reply_to_terminate_agent(self, messages=None, sender=None, config=None):
        return node(True), node({"content": "TERMINATE"})

    def _reply_to_terminate_extractor(self, messages=None, sender=None, config=None):
        return node(True), node({"content": "TERMINATE"})


poem_agent = PoemAgent(silent=True)
env = llfbench.make("llf-poem-SyllableConstrainedPoem-v0", instruction_type="b", feedback_type="a")

# ======= Now with the env reward, we can optimize =======

optimizer = TeacherLLMOptimizer(
    poem_agent.student_agent.parameters,
    config_list=config_list,
    task_description=dedent(
        """
                         You are helping a student write a poem that satisfies the requirement of having a fixed
                         number of syllables per line and a fixed number of lines.
                         """
    ),
)

performances = []
exp_runs = 5
optimization_steps = 3

for _ in range(exp_runs):
    info = train_with_env(env, poem_agent, optimizer, optimization_steps, feedback_verbalize=verbalize, verbose=True)
    print("Agent reward history:", info["rewards"])
    performances.append(info["rewards"])

performances = backfill_lists(performances)
plot_agent_performance(performances, backfilled=True)