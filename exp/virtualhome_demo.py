import sys
import os

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

def env_fn(env_id):
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

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@dataclass
class Config:
    agent_num=2
    max_episode_length=250
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
    debug=False

class TraceArena(object):
    def __init__(self, max_number_steps, arena_id, env_fn, agent_fn, record_dir='out',
                 debug=False, run_predefined_actions=False, comm=False, args=None):
        # skipped some initialization

        print("Init Env")
        self.env = env_fn(arena_id)
        self.converse_agents = []
        self.comm_cost_info = {
            "converse": {"overall_tokens": 0, "overall_times": 0},
            "select": {"tokens_in": 0, "tokens_out": 0}
        }

        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.run_predefined_actions = run_predefined_actions
        atexit.register(self.close)

        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}

        self.arena_id = arena_id
        self.env_fn = env_fn
        self.agent_names = ["Agent_{}".format(i+1) for i in range(len(agent_fn))]

    def close(self):
        self.env.close()

    def get_port(self):
        return self.env.port_number

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

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
            agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name, self.env.room_info, self.env.goal_spec[it])

    def get_actions(self, obs, action_space=None, true_graph=False):
        dict_actions = {}

        # allow discussion
        if self.comm:
            pass

        logger.info('Actions at step {}'.format(self.env.steps))
        for it, agent in enumerate(self.agents):
            if self.task_goal is None:
                goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
            else:
                goal_spec = self.env.get_goal(self.task_goal[it], self.env.agent_goals[it])

            # for the LLM agent
            obs = self.env.get_observations()
            # TODO: how should we handle dialog history?
            dict_actions[it], self.dict_info[it] = agent.get_action(obs[it], goal_spec,
                                                                    dialogue_history=self.dict_dialogue_history[
                                                                        self.agent_names[it]])

        return dict_actions, self.dict_info

    def step(self, true_graph=False):
        if self.env.steps == 0:
            pass
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space, true_graph=true_graph)

        for i in range(len(dict_info)):
            if len(dict_info) > 1 and 'subgoals' in dict_info[i]:
                dup = self.env.check_subgoal(dict_info[i]['subgoals'])
                self.cnt_nouse_subgoal += dup
                if i == 0 and 'subgoals' in dict_info[i + 1].keys() and dict_info[i]['subgoals'] == dict_info[i + 1]['subgoals']:
                    self.cnt_duplicate_subgoal += 1
        try:
            step_info = self.env.step(dict_actions)
        except Exception as e:
            print("Exception occurs when performing action: ", dict_actions)
            raise Exception
        return step_info, dict_actions, dict_info

    def run(self, random_goal=False, pred_goal=None, cnt_subgoal_info = False):
        # run till the end of episode...
        while True:
            (obs, reward, done, infos, messages), actions, agent_info = self.step()
        pass


"""
Goal:
High-level wrappers to take steps in the environment
Return text observations for each agent, and reward and stuff
"""

class TraceVirtualHome:
    def __init__(self, max_number_steps, run_id, env_fn, agent_fn, num_agents, record_dir='out', debug=False, run_predefined_actions=False, comm=False, args=None):
        print("Init Env")
        self.env = env_fn(run_id)
        self.converse_agents = []
        self.comm_cost_info = {
            "converse": {"overall_tokens": 0, "overall_times": 0},
            "select": {"tokens_in": 0, "tokens_out": 0}
        }

        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.run_predefined_actions = run_predefined_actions
        atexit.register(self.close)

        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}

        self.arena_id = run_id
        self.env_fn = env_fn
        self.agent_names = ["Agent_{}".format(i + 1) for i in range(len(agent_fn))]

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

        env = env_fn(run_id)
        self.env = env
        self.agents = []
        self.num_agents = num_agents

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
        agent_obs = {}

        for it in range(self.num_agents):
            goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
            agent_goal_specs[it] = goal_spec

            agent_obs[it] = self.agents[it].get_obs_forLLM_plan()

        # use these two, we can generate plans...
        return agent_obs, agent_goal_specs

    def close(self):
        self.env.close()

    def get_port(self):
        return self.env.port_number

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

    def env_step(self, dict_actions):
        try:
            step_info = self.env.step(dict_actions)
        except Exception as e:
            print("Exception occurs when performing action: ", dict_actions)
            raise Exception

        return step_info
    def step(self, plans, a_infos, LM_times):
        """
        plans: {agent_id: plan}
        a_infos: {agent_id: info}
        LM_times: {agent_id: time}

        Get raw obs from env, turn into text obs
        """
        if self.env.steps == 0:
            pass

        dict_actions = {}

        for it, agent in enumerate(self.agents):
            dict_actions[it], self.dict_info[it] = agent.get_action(plans[it], a_infos[it], LM_times[it],
                                                                    dialogue_history=self.dict_dialogue_history[
                                                                        self.agent_names[it]])

        step_info = self.env_step(dict_actions)

        return step_info, dict_actions, self.dict_info


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

if __name__ == '__main__':

    args = Config()



