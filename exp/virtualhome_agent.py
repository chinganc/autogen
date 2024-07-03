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
import logging
from dataclasses import dataclass

import atexit
from collections import defaultdict

import argparse

from collections import defaultdict

import random
import json
from json import JSONDecodeError
import pandas as pd
import backoff
import logging

logger = logging.getLogger("__main__")


class LLM_agent:
    """
    LLM agent class
    """

    def __init__(self, agent_id, args):
        self.debug = args.debug
        self.agent_type = 'LLM'
        self.agent_names = ["Agent_{}".format(i + 1) for i in range(args.agent_num)]
        self.teammate_names = self.agent_names[:agent_id - 1] + self.agent_names[agent_id:]
        self.agent_id = agent_id
        self.teammate_agent_id = [i + 1 for i in range(args.agent_num)]
        self.teammate_agent_id.remove(agent_id)
        # self.lm_id = args.lm_id_list[agent_id - 1]
        # self.llm_config = args.llm_config_list[agent_id - 1]
        self.prompt_template_path = args.prompt_template_path
        self.args = args
        self.LLM = LLM(self.prompt_template_path, self.args, self.agent_id, self.agent_names)

        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}
        self.steps = 0
        self.plan = None
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None

        self.teammate_grabbed_objects = defaultdict(list)
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}
        self.id_inside_room = {}
        self.satisfied = []
        self.reachable_objects = []
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }

    @property
    def all_relative_name(self) -> list:
        return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']

    def goexplore(self, LM_times):
        target_room_id = int(self.plan.split(' ')[-1][1:-1])
        if self.current_room['id'] == target_room_id:
            self.plan = None
            return None
        elif self.current_room['class_name'] != self.last_room[
            'class_name'] and LM_times == 0:  # just entered a new room, check if the agent should stop here
            self.plan = None
            return None
        return self.plan.replace('[goexplore]', '[walktowards]')

    def gocheck(self):
        assert len(self.grabbed_objects) < 2  # must have at least one free hands
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None
        if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]')  # conflict will work right?
        else:
            return self.plan.replace('[gocheck]', '[walktowards]')

    def gograb(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None
        assert len(self.grabbed_objects) < 2  # must have at least one free hands

        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[
            target_object_room]] or target_object_id in [x['id'] for x_list in self.teammate_grabbed_objects.values()
                                                         for x in x_list]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walktowards]')

    def goput(self):
        # if len(self.progress['goal_location_room']) > 1: # should be ruled out
        if len(self.grabbed_objects) == 0:
            self.plan = None
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"

    # def LLM_plan(self):
    #     if len(self.grabbed_objects) == 2:
    #         return f"[goput] {self.goal_location}", {}
    #
    #     return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied,
    #                         self.unchecked_containers,
    #                         self.ungrabbed_objects, self.id_inside_room[self.goal_location_id],
    #                         self.action_history, self.dialogue_history, self.teammate_grabbed_objects,
    #                         [self.id_inside_room[teammate_agent_id_item] for teammate_agent_id_item in
    #                          self.teammate_agent_id], None, self.steps)

    def get_obs_forLLM_plan(self):

        # this is hard-coded by the github (not by us)
        # basically saying once we have two items, we hard code goput
        # but maybe this is unnecessary
        # also this should happen on the agent level...

        # if len(self.grabbed_objects) == 2:
        #     return f"[goput] {self.goal_location}", {}

        return self.LLM.get_runnable(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied,
                                     self.unchecked_containers,
                                     self.ungrabbed_objects, self.id_inside_room[self.goal_location_id],
                                     self.action_history, self.dialogue_history, self.teammate_grabbed_objects,
                                     [self.id_inside_room[teammate_agent_id_item] for teammate_agent_id_item in
                                      self.teammate_agent_id],
                                     self.id_inside_room, self.teammate_agent_id,
                                     None, self.steps)

    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and \
                        id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
                # if self.debug:
                # 	print(satisfied)
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied

    def filter_graph(self, obs):
        relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
        relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
        new_graph = {
            "edges": [edge for edge in obs['edges'] if
                      edge['from_id'] in relative_id and edge['to_id'] in relative_id],
            "nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
        }

        return new_graph

    def obs_processing(self, observation, goal):
        satisfied, unsatisfied = self.check_progress(observation, goal)
        # print(f"satisfied: {satisfied}")
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied
        obs = self.filter_graph(observation)
        self.grabbed_objects = []
        teammate_grabbed_objects = defaultdict(list)
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in obs['nodes']}
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]
                elif r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            # elif x == self.teammate_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
            # 	teammate_grabbed_objects.append(self.id2node[y])
            else:
                for i, teammate_agent_id_item in enumerate(self.teammate_agent_id):
                    if x == teammate_agent_id_item and r in ['HOLDS_RH', 'HOLDS_LH']:
                        teammate_grabbed_objects[self.teammate_names[i]].append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []
        # print(teammate_grabbed_objects)
        # print([w['id'] for w_list in teammate_grabbed_objects.values() for w in w_list])
        for x in obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w_list in teammate_grabbed_objects.values()
                                                              for w in w_list]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all(
                    [x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x[
                'id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w_list in
                                                                      teammate_grabbed_objects.values() for w in
                                                                      w_list]:
                ungrabbed_objects.append(x)

        if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in \
                self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': obs,
                "obs": {
                    "grabbed_objects": self.grabbed_objects,
                    "teammate_grabbed_objects": teammate_grabbed_objects,
                    "reachable_objects": self.reachable_objects,
                    "progress": {
                        "unchecked_containers": self.unchecked_containers,
                        "ungrabbed_objects": self.ungrabbed_objects,
                    },
                    "satisfied": self.satisfied,
                    "current_room": self.current_room['class_name'],
                },
                }
        for i, teammate_agent_id_item in enumerate(self.teammate_agent_id):
            if self.id_inside_room[teammate_agent_id_item] == self.current_room['class_name']:
                self.teammate_grabbed_objects[self.teammate_names[i]] = teammate_grabbed_objects[self.teammate_names[i]]

        return info

    def progress2text(self):
        return self.LLM.progress2text(self.current_room, [self.id2node[x] for x in self.grabbed_objects],
                                      self.unchecked_containers, self.ungrabbed_objects,
                                      self.id_inside_room[self.goal_location_id], self.satisfied,
                                      self.teammate_grabbed_objects,
                                      [self.id_inside_room[teammate_agent_id_item] for teammate_agent_id_item in
                                       self.teammate_agent_id], None, self.steps)

    def get_action(self, plan, a_info, LM_times, observation, goal, dialogue_history=None):
        """
        a_info: passed in from LLM module's get_runnable()

        takes in a plan, a context (how many times LLM has been called, and execute the plan into action)

        :param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
        "nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
        "messages": [None, None]
        }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        # Apply autogen as an independant communication module
        if self.args.comm and dialogue_history is not None and dialogue_history != []:
            self.dialogue_history = [word for dialogue in dialogue_history for word in dialogue]

        info = self.obs_processing(observation, goal)
        action = None

        if plan is None or plan == 'None':
            print("No more things to do!")
            plan = f"[wait]"
            action = None

        self.plan = plan
        a_info.update({"steps": self.steps})
        info.update({"LLM": a_info})

        if self.plan.startswith('[goexplore]'):
            action = self.goexplore(LM_times)
            # if already in the room
            # plan is set to None
            # action is set to None
        elif self.plan.startswith('[gocheck]'):
            action = self.gocheck()
            # if the container is already open
            # then action, and plan will both be None
        elif self.plan.startswith('[gograb]'):
            action = self.gograb()
        elif self.plan.startswith('[goput]'):
            action = self.goput()
        elif self.plan.startswith('[send_message]'):
            action = self.plan[:]
            self.plan = None
        elif self.plan.startswith('[wait]'):
            action = None
        elif self.plan.startswith('[walktowards]'):
            action = self.plan
        else:
            raise ValueError(f"unavailable plan {self.plan}, action: {action}")

        if self.plan is None and action is None:
            # this is actually a hidden "retry" condition from the original code
            # we shouldn't be here if we explicitly add a pre-check (action needs to come from the available actions)

            # one condition is -- if the agent enters a new room, goexplore force the agent to wait one turn
            self.plan = f"[wait]"
            action = None
            # import pdb; pdb.set_trace()

        if action is not None:
            if action.startswith('[send_message]'):
                self.action_history.append(action.split(":")[0])
            else:
                self.action_history.append(action if action is not None else self.plan)
        else:
            self.action_history.append(action if action is not None else self.plan)

        self.steps += 1
        info.update({"plan": self.plan})
        if action == self.last_action and self.current_room['class_name'] == self.last_room['class_name']:
            self.stuck += 1
        else:
            self.stuck = 0

        self.last_action = action
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0

        return action, info

    # def get_action(self, observation, goal, dialogue_history=None):
    #     """
    #     :param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
    #     "nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
    #     "messages": [None, None]
    #     }
    #     :param goal:{predicate:[count, True, 2]}
    #     :return:
    #     """
    #     # Apply autogen as an independant communication module
    #     if self.args.comm and dialogue_history is not None and dialogue_history != []:
    #         self.dialogue_history = [word for dialogue in dialogue_history for word in dialogue]
    #
    #     info = self.obs_processing(observation, goal)
    #     action = None
    #     LM_times = 0
    #     while action is None:
    #         if self.plan is None:
    #             if LM_times > 0:
    #                 print("Retrying LM Plan: ", info)
    #                 self.LLM.llm_config.update({"seed": random.randint(0, 100000)})
    #             if LM_times > 3:
    #                 plan = f"[wait]"  # TODO just workround
    #             # raise Exception(f"retrying LM_plan too many times")
    #             plan, a_info = self.LLM_plan()
    #             # if self.debug:
    #             # 	print(LM_times, "plan: ", plan)
    #             if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
    #                 print("No more things to do!")
    #                 plan = f"[wait]"
    #             self.plan = plan
    #             a_info.update({"steps": self.steps})
    #             info.update({"LLM": a_info})
    #             LM_times += 1
    #         if self.plan.startswith('[goexplore]'):
    #             action = self.goexplore(LM_times)
    #         elif self.plan.startswith('[gocheck]'):
    #             action = self.gocheck()
    #         elif self.plan.startswith('[gograb]'):
    #             action = self.gograb()
    #         elif self.plan.startswith('[goput]'):
    #             action = self.goput()
    #         elif self.plan.startswith('[send_message]'):
    #             action = self.plan[:]
    #             self.plan = None
    #         elif self.plan.startswith('[wait]'):
    #             action = None
    #             break
    #         else:
    #             raise ValueError(f"unavailable plan {self.plan}")
    #
    #     self.action_history.append(action if action is not None else self.plan)
    #     # if self.debug:
    #     # print("action:\n", action, "\n\n")
    #     # logger.info(f"action:\n{action}\n")
    #     self.steps += 1
    #     info.update({"plan": self.plan,
    #                  })
    #     if action == self.last_action and self.current_room['class_name'] == self.last_room['class_name']:
    #         self.stuck += 1
    #     else:
    #         self.stuck = 0
    #     self.last_action = action
    #     self.last_room = self.current_room
    #     if self.stuck > 20:
    #         print("Warning! stuck!")
    #         self.action_history[-1] += ' but unfinished'
    #         self.plan = None
    #         if type(self.id_inside_room[self.goal_location_id]) is list:
    #             target_room_name = self.id_inside_room[self.goal_location_id][0]
    #         else:
    #             target_room_name = self.id_inside_room[self.goal_location_id]
    #         action = f"[walktowards] {self.goal_location}"
    #         if self.current_room['class_name'] != target_room_name:
    #             action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
    #         self.stuck = 0
    #
    #     return action, info

    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        self.steps = 0
        self.containers_name = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name = rooms_name
        self.roomname2id = {x['class_name']: x['id'] for x in room_info}
        self.id2node = {x['id']: x for x in obs['nodes']}
        self.stuck = 0
        self.last_room = None
        self.unsatisfied = {k: v[0] for k, v in goal.items()}
        self.satisfied = []
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.id_inside_room = {self.goal_location_id: self.rooms_name[:]}
        for teammate_agent_id_item in self.teammate_agent_id:
            self.id_inside_room[teammate_agent_id_item] = None
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.teammate_grabbed_objects = defaultdict(list)
        if self.debug:
            print(self.agent_id)
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            # print(x, r, y)
            if x == self.agent_id and r == 'INSIDE':
                self.current_room = self.id2node[y]
                self.last_room = self.current_room
        self.plan = None
        self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)


import random
from autogen import oai
import openai
import torch
import json
from json import JSONDecodeError
import os
import pandas as pd
# from openai.error import OpenAIError
from openai import OpenAIError

import backoff
import logging

logger = logging.getLogger("__main__")


class ChatCompletionManager:
    def __init__(self):
        self.human_agents = {}
        self.next_port = 7861

    def close(self):
        for port, item in self.human_agent_dict.items():
            # Close the parent connection
            item[2].join()
        for port, item in self.human_agent_dict.items():
            item[0].close()

    def create(self, messages, config_list=None, *args, **kwargs):
        if kwargs.get('human_agent') is None:
            import autogen
            return autogen.OpenAIWrapper(config_list=config_list).create(messages=messages, *args, **kwargs)

            # return oai.ChatCompletion.create(messages=messages, config_list=config_list, request_timeout=600, *args, **kwargs)

        # human player
        else:
            port = kwargs['human_agent']
            parent_conn, child_conn, _ = self.human_agents[port]
            str_message = ""
            for i in range(len(messages)):
                str_message += f"{messages[i]['role']} :" + messages[i]['content'] + '\n'

            parent_conn.send(str_message)
            str_result = parent_conn.recv()
            print("str_result:", str_result)

            if '{' not in str_result:
                str_result = '{"action": "' + str_result + '", "thoughts": ""}'

            # json_result = json.loads(str_result)
            # if json_result.get('thoughts') is None:
            #     json_result['thoughts'] = ""

            # str_result = json.dumps(json_result)

            result = {
                'usage': {
                    'prompt_tokens': 9999,
                    'completion_tokens': 9999,
                },
                'choices': [{'message': {'content': str_result, 'role': 'assistant'}}]
            }
            return result


# oai_wrapper = ChatCompletionManager()

class LLM:
    def __init__(self,
                 prompt_template_path,
                 args,
                 agent_id,
                 agent_names
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_names = agent_names
        self.agent_name = self.agent_names[agent_id - 1]
        self.teammate_names = self.agent_names[:agent_id - 1] + self.agent_names[agent_id:]
        if self.teammate_names == []:
            self.teammate_names = ['nobody']
        self.teammate_pronoun = "she"
        self.debug = args.debug
        self.log_thoughts = args.log_thoughts
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.organization_instructions = args.organization_instructions
        self.action_history_len = args.action_history_len
        self.dialogue_history_len = args.dialogue_history_len
        if self.prompt_template_path is not None:
            self.single = 'single' in self.prompt_template_path or self.teammate_names == ['nobody']
            df = pd.read_csv(self.prompt_template_path)
            self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$TEAMMATE_NAME$",
                                                                                                    ", ".join(
                                                                                                        self.teammate_names))

        # self.lm_id = lm_id
        # self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id or 'chat' in lm_id or 'human' in lm_id
        self.total_cost = 0
        self.all_actions = 0
        self.failed_actions = 0

    def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)

    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
            elif relation == 'sit':
                continue
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"

    def parse_answer(self, available_actions, text):
        self.all_actions += 1
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(
                    ' ') or f"Option {option}" in text or f"({option})" in text or option == text[0]:
                return action
        self.failed_actions += 1
        if self.debug:
            logger.info(f"Agent_{self.agent_id} failed to generate actions: {self.failed_actions}/{self.all_actions}")
            logger.info("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)

    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room,
                      satisfied, teammate_grabbed_objects, teammate_last_room, room_explored, steps):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        ### teammate modeling
        if not self.single:
            for i, teammate_name in enumerate(self.teammate_names):
                ss = ""
                if len(teammate_grabbed_objects[teammate_name]) == 0:
                    ss += "nothing. "
                else:
                    ss += f"<{teammate_grabbed_objects[teammate_name][0]['class_name']}> ({teammate_grabbed_objects[teammate_name][0]['id']}). "
                    if len(teammate_grabbed_objects[teammate_name]) == 2:
                        ss = ss[
                             :-2] + f" and <{teammate_grabbed_objects[teammate_name][1]['class_name']}> ({teammate_grabbed_objects[teammate_name][1]['id']}). "
                if teammate_last_room[i] is None:
                    s += f"I don't know where {teammate_name} is. "
                elif teammate_last_room[i] == current_room['class_name']:
                    s += f"I also see {teammate_name} here in the {current_room['class_name']}, {self.teammate_pronoun} is holding {ss}"
                else:
                    s += f"Last time I saw {teammate_name} was in the {teammate_last_room[i]}, {self.teammate_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        return f"This is step {steps}. " + s

    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored,
                            id_inside_room, teammate_agent_id, current_room):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        """
        available_plans = []
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")

        # added [wait] to always be an option
        available_plans.append('[wait]')

        # check if the other agent is in the same room, if so, add the plan to send message
        if not self.single:
            for i, teammate_name in enumerate(self.teammate_names):
                if id_inside_room[teammate_agent_id[i]] == current_room['class_name']:
                    available_plans.append(f"[send_message] <{teammate_name}> ({teammate_agent_id[i]}): Replace the text after colon as the actual message you want to send.")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"


        return plans, len(available_plans), available_plans

    def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room,
            action_history, dialogue_history, teammate_grabbed_objects, teammate_last_room,
            id_inside_room, teammate_agent_id,
            room_explored=None,
            steps=None):
        info = {}
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects,
                                           goal_location_room, satisfied, teammate_grabbed_objects, teammate_last_room,
                                           room_explored, steps)
        action_history_desc = ", ".join(action_history[-self.action_history_len:] if len(
            action_history) > self.action_history_len else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-self.dialogue_history_len:] if len(
            dialogue_history) > self.dialogue_history_len else dialogue_history)
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        if self.organization_instructions is not None:
            prompt = prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        message = None
        info.update({"goal": self.goal_desc,
                     "progress": progress_desc,
                     "action_history": action_history_desc,
                     "dialogue_history_desc": dialogue_history_desc})
        prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers,
                                                                              ungrabbed_objects, message, room_explored,
                                                                              id_inside_room, teammate_agent_id,
                                                                              current_room)
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                         "plan": None})
            return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        # if steps == 0:
        #     print(f"base_prompt:\n{prompt}\n\n")
        # if self.debug:
        #     logger.info(f"base_prompt:\n{prompt}\n\n")
        # for _ in range(3):
        #     try:
        #         outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt,
        #                                         self.sampling_params)
        #         if outputs[0] != '{' and '{' in outputs:
        #             outputs = '{' + outputs.split('{')[1].strip()
        #             outputs = outputs.split('}')[0].strip() + '}'
        #         outputs_json = json.loads(outputs)
        #         output = outputs_json["action"]
        #         thoughts = outputs_json["thoughts"]
        #         break
        #     except (JSONDecodeError, KeyError) as e:
        #         print(outputs)
        #         print(e)
        #         self.llm_config.update({"seed": self.llm_config["seed"] + 1})
        #         self.generator = self.lm_engine()
        #         print("retrying...")
        #
        #         continue

        # info['cot_usage'] = usage

        # logger.info(f"action_output: {output}")
        # if self.log_thoughts:
        #     logger.info(f"thoughts: {thoughts}")

        plan = self.parse_answer(available_plans_list, output)

        # if self.debug:
        #     logger.info(f"plan:\n{plan}")
        info.update({"num_available_actions": num,
                     "prompts": prompt,
                     "outputs": outputs,
                     "plan": plan,
                     "total_cost": self.total_cost})

        return plan, info

    def get_runnable(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects,
                     goal_location_room,
                     action_history, dialogue_history, teammate_grabbed_objects, teammate_last_room,
                     id_inside_room, teammate_agent_id,
                     room_explored=None,
                     steps=None):
        info = {}
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects,
                                           goal_location_room, satisfied, teammate_grabbed_objects, teammate_last_room,
                                           room_explored, steps)
        action_history_desc = ", ".join(action_history[-self.action_history_len:] if len(
            action_history) > self.action_history_len else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-self.dialogue_history_len:] if len(
            dialogue_history) > self.dialogue_history_len else dialogue_history)
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        if self.organization_instructions is not None:
            prompt = prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        message = None
        info.update({"goal": self.goal_desc,
                     "progress": progress_desc,
                     "action_history": action_history_desc,
                     "dialogue_history_desc": dialogue_history_desc})
        prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers,
                                                                              ungrabbed_objects, message, room_explored,
                                                                              id_inside_room, teammate_agent_id, current_room)
        info['available_plans'] = available_plans_list
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                         "plan": None})
            return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        info.update({"num_available_actions": num,
                     "prompts": prompt})

        return info

# write a Trace agent
# a few ideas:
# 1. Need to decide who to send message to, based on previous messages
# 2. Need to act

# communicate
# - choose_person (a list of agents, past history, etc.)
# - send_message
# - terminate

# act

# decentralized action/control, if the agent outputs [wait], then we
