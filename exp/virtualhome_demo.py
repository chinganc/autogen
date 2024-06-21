import sys
import os

import re

# manually append the path
sys.path.append("/piech/u/anie/Organized-LLM-Agents/envs/cwah")
sys.path.append("/piech/u/anie/Organized-LLM-Agents/envs")
sys.path.append("/piech/u/anie/autogen/")

import subprocess

# Command to get the process IDs using lsof and kill them
command = "kill -9 $(lsof -t -i :6314)"

# Run the command
subprocess.run(command, shell=True)

import pickle
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from envs.unity_environment import UnityEnvironment
# from agents import LLM_agent
# from arguments import get_args
# from algos.arena_mp2 import ArenaMP
import logging
from dataclasses import dataclass

import virtualhome_agent

import atexit
from collections import defaultdict

import argparse


def env_fn(env_id, env_task_set, executable_args, args):
    return UnityEnvironment(num_agents=args.agent_num,
                            max_episode_length=args.max_episode_length,
                            port_id=env_id,
                            env_task_set=env_task_set,
                            agent_goals=['LLM' for i in range(args.agent_num)],
                            observation_types=[args.obs_type for i in range(args.agent_num)],
                            use_editor=args.use_editor,
                            executable_args=executable_args,
                            base_port=args.base_port if args.base_port is not None else np.random.randint(11000, 13000),
                            seed=args.seed,
                            recording_options={'recording': True if args.gen_video else False,
                                               'output_folder': args.record_dir,
                                               'file_name_prefix': args.mode,
                                               'cameras': 'PERSON_FROM_BACK',
                                               'modality': 'normal'}
                            )


"""
Goal:
High-level wrappers to take steps in the environment
Return text observations for each agent, and reward and stuff

Check:
1. Write the communication function
"""


class TraceVirtualHome:
    def __init__(self, max_number_steps, run_id, env_fn, agent_fn, num_agents, record_dir='out', debug=False,
                 run_predefined_actions=False, comm=False, args=None):
        print("Init Env")
        self.converse_agents = []
        self.comm_cost_info = {
            "converse": {"overall_tokens": 0, "overall_times": 0},
            "select": {"tokens_in": 0, "tokens_out": 0}
        }

        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}

        self.arena_id = run_id
        self.env_fn = env_fn
        self.agent_names = ["Agent_{}".format(i + 1) for i in range(len(agent_fn))]

        self.prompt_template_path = args.prompt_template_path
        self.organization_instructions = args.organization_instructions

        env_task_set = pickle.load(open(args.dataset_path, 'rb'))
        logging.basicConfig(format='%(asctime)s - %(name)s:\n %(message)s', level=logging.INFO)
        logger = logging.getLogger(__name__)

        # skipped a lot of logging

        args.record_dir = f'../test_results/{args.mode}'
        logger.info("mode: {}".format(args.mode))
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)

        if "image" in args.obs_type or args.gen_video:
            os.system("Xvfb :94 & export DISPLAY=:94")
            import time
            time.sleep(3)  # ensure Xvfb is open
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            executable_args = {
                'file_name': args.executable_file,
                'x_display': '94',
                'no_graphics': False,
                'timeout_wait': 5000,
            }
        else:
            executable_args = {
                'file_name': args.executable_file,
                'no_graphics': True,
                'timeout_wait': 500,
            }

        self.executable_args = executable_args
        self.args = args
        self.env_task_set = env_task_set

        env = env_fn(run_id, env_task_set, executable_args, args)
        self.env = env
        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.run_predefined_actions = run_predefined_actions
        atexit.register(self.close)

        self.agents = []
        self.num_agents = num_agents

        self.dialogue_history_len = 30

        for i in range(self.num_agents):
            self.agents.append(virtualhome_agent.LLM_agent(agent_id=i + 1, args=args))

    def reset(self, task_id=None, reset_seed=None):
        self.cnt_duplicate_subgoal = 0
        self.cnt_nouse_subgoal = 0
        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}
        self.converse_agents = []
        self.comm_cost_info = {
            "converse": {"overall_tokens": 0, "overall_times": 0},
            "select": {"tokens_in": 0, "tokens_out": 0}
        }
        for i in range(self.num_agents):
            # reset conversable agents
            pass

        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id, reset_seed=reset_seed)

        for it, agent in enumerate(self.agents):
            agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name,
                        self.env.room_info, self.env.goal_spec[it])

        # return observation required for planning
        obs = self.env.get_observations()
        agent_goal_specs = {}
        agent_goal_descs = {}
        agent_infos = {}
        agent_obs_descs = {}
        agent_obs = obs

        for it in range(self.num_agents):
            goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
            agent_goal_specs[it] = goal_spec
            info = self.agents[it].obs_processing(obs[it], goal_spec)

            agent_obs_descs[it] = self.agents[it].get_obs_forLLM_plan()
            agent_infos[it] = info
            agent_goal_descs[it] = agent.LLM.goal_desc

        # use these two, we can generate plans...
        return agent_obs, agent_obs_descs, agent_goal_specs, agent_goal_descs, agent_infos

    def close(self):
        self.env.close()

    def get_port(self):
        return self.env.port_number

    def reset_env(self):
        self.env.close()
        self.env = env_fn(self.arena_id, self.env_task_set, self.executable_args, self.args)

    def env_step(self, dict_actions):
        try:
            step_info = self.env.step(dict_actions)
        except Exception as e:
            print("Exception occurs when performing action: ", dict_actions)
            raise Exception

        return step_info

    def prep_discussion(self, agent_obs):
        df = pd.read_csv(self.prompt_template_path)
        prompt_format = df['prompt'][1]
        selector_prompts = []

        # init / truncate dialogue_history
        for it, agent in enumerate(self.agents):
            if self.dict_dialogue_history[self.agent_names[it]] != []:
                if len(self.dict_dialogue_history[self.agent_names[it]]) > self.dialogue_history_len:
                    self.dict_dialogue_history[self.agent_names[it]] = self.dict_dialogue_history[self.agent_names[it]][
                                                                       -self.dialogue_history_len:]
                dialogue_history = [word for dialogue in self.dict_dialogue_history[self.agent_names[it]] for word in
                                    dialogue]
            else:
                dialogue_history = []

            goal_desc = agent.LLM.goal_desc
            action_history = self.agents[it].action_history
            action_history = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

            goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])

            _ = self.agents[it].obs_processing(agent_obs[it], goal_spec)
            progress = self.agents[it].progress2text()

            teammate_names = self.agent_names[:it] + self.agent_names[it + 1:]
            selector_prompt = prompt_format.replace("$AGENT_NAME$", self.agent_names[it])
            selector_prompt = selector_prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
            selector_prompt = selector_prompt.replace("$TEAMMATE_NAME$", ", ".join(teammate_names))
            selector_prompt = selector_prompt.replace("$GOAL$", str(goal_desc))
            selector_prompt = selector_prompt.replace("$PROGRESS$", str(progress))
            selector_prompt = selector_prompt.replace("$ACTION_HISTORY$", str(action_history))
            selector_prompt = selector_prompt.replace("$DIALOGUE_HISTORY$", str('\n'.join(dialogue_history)))
            selector_prompts.append(selector_prompt)

        return selector_prompts

    def communicate(self, agents):
        # 3 agents
        # each of them sees own observation
        # send_to_agent(agent1), return {agent_name, message_to_send}
        # 1 round of message sending.
        pass

    def step(self, plans, agent_infos, LM_times, agent_obs, agent_goal_specs,
             prev_agent_obs_desc, sticky_action=True):
        """
        The only thing we need from Trace agent is
        plans and LM_times

        plans: {agent_id: plan} (can be generated by a trace agent)
        agent_infos: {agent_id: info}
        LM_times: {agent_id: time}
        dict_dialogue_history: {agent_name: dialogue_history}

        Get raw obs from env, turn into text obs
        and reward, and terminate

        The prompt will change because it will say it's the step 2 or something...
        """
        if self.env.steps == 0:
            pass

        dict_actions = {}

        # self.dict_dialogue_history[self.agent_names[it]]

        # if neither agent's observation text is changing, we execute the same action
        # until it changes (sticky_action)

        # TODO: need to produce feedback / reward if the plan/action is invalid
        # but this is not the most major issue because invalid actions shouldn't be listed as an action

        # TODO: find where the actual target is...and simulate a success agent (as human)
        # TODO: to test if this interface works end-to-end

        # we set it to -1 reward if the action is invalid
        max_sticky_steps = 10
        while max_sticky_steps > 0:
            for it, agent in enumerate(self.agents):
                dict_actions[it], self.dict_info[it] = agent.get_action(plans[it], agent_infos[it], LM_times[it],
                                                                        agent_obs[it], agent_goal_specs[it],
                                                                        dialogue_history=self.dict_dialogue_history)
            step_info = self.env_step(dict_actions)
            # determine if both agents are seeing the same

            #  (obs, reward, done, infos, messages) = step_info
            # obs is None by default for Unity env
            obs = self.env.get_observations()
            step_info = [obs, step_info[1], step_info[2], step_info[3], step_info[4]]
            agent_obs, reward, done, infos, messages = step_info
            if done:
                break

            # we explicitly process it before generating the description
            for it in range(len(self.agents)):
                self.agents[it].obs_processing(agent_obs[it], agent_goal_specs[it])

            agent_obs_descs = {}
            for it in range(len(self.agents)):
                agent_obs_descs[it] = self.agents[it].get_obs_forLLM_plan()

            # compare progress text with the original one (where we started)
            # compare the available actions
            # if both are the same, we continue without getting back to the LLM Agent
            break_loop = False
            for it in range(len(self.agents)):
                text = prev_agent_obs_desc[it]['prompts']
                pattern = r'(step \d+)'
                text = re.sub(pattern, f'step X', text)
                pattern = r'Progress:.*|Available actions:.*(?:\n.*\w.*)+'
                matches = re.findall(pattern, text)

                new_text = agent_obs_descs[it]['prompts']
                pattern = r'(step \d+)'
                new_text = re.sub(pattern, f'step X', new_text)
                pattern = r'Progress:.*|Available actions:.*(?:\n.*\w.*)+'
                new_matches = re.findall(pattern, new_text)

                print("obs progress: ", new_matches[0])

                if matches != new_matches:
                    break_loop = True
                    break

            if break_loop:
                break

            max_sticky_steps -= 1
            print("Performing the same action again...stikcy counter is: ", max_sticky_steps)

        return step_info, agent_obs_descs, dict_actions, self.dict_info


# 3 agents
# each of them sees own observation
# send_to_agent(agent1), return {agent_name, message_to_send}
# 1 round of message sending.

# static one
# can call the OAI to select the agent

# simple version -- optimize the prompt

"""
1. An arena to coordinate all agents, reset all of them, take centralized step
2. An agent that can send prompts
"""

"""
Write a high-level wrapper of an environment
that supports high-level actions
([goexplore], ...)

**Build the high-level gym Env**
"""

"""
@trace.bundle()
def act():
    #plan()
    #discuss()
"""

"""
Steps:
Let's first have an agent that just acts...without communication

Q: when does "send_message" occur as an option for the agent?
"""


class RandomAgent:
    def __init__(self):
        pass

    def extract_available_actions(self, text):
        # Define the regex pattern to extract available actions without the letter
        pattern = r'[A-Z]\. (.+)'

        # Find all matches
        matches = re.findall(pattern, text)

        return matches

    def act(self, obs):
        # obs: text observation
        available_actions = self.extract_available_actions(obs)
        return random.choice(available_actions)


def rollout(env, agent, config, horizon=50):
    # plans = {
    #     0: "[gocheck] <kitchencabinet> (74)",
    #     1: "[goexplore] <kitchen> (172)"
    # }

    LM_times = {
        0: 0,
        1: 0
    }

    agent_obs, agent_obs_descs, agent_goal_specs, agent_goal_descs, agent_infos = env.reset()
    rewards = []

    for _ in range(horizon):
        plans = agent.act(agent_obs_descs)
        step_info, agent_obs_descs, dict_actions, dict_info = env.step(plans, agent_infos, LM_times, agent_obs,
                                                                       agent_goal_specs, agent_obs_descs)
        agent_obs, reward, done, infos, messages = step_info
        rewards.append(reward)

        if done:
            break

    return rewards


@dataclass
class Config:
    agent_num = 2
    max_episode_length = 250
    use_editor = False
    base_port = None
    seed = 1
    gen_video = True
    record_dir = ""
    mode = 'test'

    dataset_path = '/piech/u/anie/Organized-LLM-Agents/envs/cwah/dataset/test_env_set_help.pik'
    obs_type = "partial"
    executable_file = "/piech/u/anie/Organized-LLM-Agents/envs/executable/linux_exec.v2.2.4.x86_64"

    log_thoughts = True
    debug = False


if __name__ == '__main__':
    args = Config()
    args.comm = False

    args.prompt_template_path = "/piech/u/anie/Organized-LLM-Agents/envs/cwah/LLM/prompt_multi_comm.csv"
    args.organization_instructions = "/piech/u/anie/Organized-LLM-Agents/envs/cwah/testing_agents/organization_instructions.csv"

    args.action_history_len = 20
    args.dialogue_history_len = 30

    max_number_steps = 250
    run_id = 0
    agent_fn = [0] * 2
    num_agents = 2

    agent = RandomAgent()
    env = TraceVirtualHome(max_number_steps, run_id, env_fn, agent_fn, num_agents, args=args)

    rollout(env, agent, args, horizon=20)
