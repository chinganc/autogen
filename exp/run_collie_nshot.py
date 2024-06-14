import autogen
from autogen.trace.nodes import node, GRAPH, ParameterNode
import string
import random
import numpy as np
from textwrap import dedent
from autogen.trace.optimizers import FunctionOptimizerV2, OPRO
from datasets import load_dataset

from typing import List
import copy
from autogen.trace.bundle import FunModule, trace_op, trace_class, TraceExecutionError
from autogen.trace.nodes import Node

import re
from tqdm import tqdm

from collie.constraints import Constraint, TargetLevel, Count, Relation

# download the data file
import os
import requests

import dill

from collie.constraint_renderer import ConstraintRenderer

"""
Best module for Collie is:
a Probe and a local modifier

Check if everything matches the constraint (Learn a verifier)
If not, propose local edit suggestions:
1. Identify the place to edit (Locate)
2. Suggest a change (Modify)

This pipeline is very easy to implement with this framework
"""

def download_file(url, file_path):
    """
    Downloads a file from the specified URL to a local file path.

    Parameters:
    - url: The URL of the file to download.
    - file_path: The local path where the file should be saved.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"File successfully downloaded to {file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def init_data():
    # URL of the file to download
    url = "https://github.com/princeton-nlp/Collie/raw/master/data/all_data.dill"
    # Local path to save the file (change this to your desired path)
    file_path = "all_data.dill"

    file_exists = os.path.exists(file_path)
    if not file_exists:
        download_file(url, file_path)


def load_data(n=500, seed=42):
    with open("all_data.dill", "rb") as f:
        all_data = dill.load(f)

    # collect all examples (with unique prompts)
    # == this is the recommended evaluation ==
    all_examples = []
    prompts = {}
    for k in all_data.keys():
        for example in all_data[k]:
            if example["prompt"] not in prompts:
                all_examples.append(example)

    random.seed(seed)
    random.shuffle(all_examples)

    return all_examples[:n]

class LLMCallable:
    def __init__(self, config_list=None, max_tokens=1024, verbose=False):
        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose

    @trace_op(catch_execution_error=False)
    def call_llm(self, user_prompt):
        """
        Sends the constructed prompt (along with specified request) to an LLM.
        """
        system_prompt = "You are a helpful assistant.\n"
        if self.verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        try:
            response = self.llm.create(
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content

        if self.verbose:
            print("LLM response:\n", response)
        return response

    def call_llm_iterate(self, user_prompt, response, feedback):
        """
        Sends the constructed prompt (along with specified request) to an LLM.
        """
        system_prompt = "You are a helpful assistant.\n"
        if self.verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback}]

        try:
            response = self.llm.create(
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content

        if self.verbose:
            print("LLM response:\n", response)
        return response

@trace_class
class Predict(LLMCallable):
    def __init__(self):
        super().__init__()

        self.demos = []
        self.prompt_template = dedent("""Please follow the constraint described in the task carefully. Here is the task:\n\n {}""")
        self.prompt_template = ParameterNode(self.prompt_template, trainable=True)

    @trace_op(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def check_and_edit_response(self, constraint, response):
        """
        We check the constraint again and see if we can improve the response.
        Our procedure should be general enough to handle different types of constraints.
        We can add individual cases to handle each constraint.
        """
        import re
        response = response.strip()
        if "with paragraphs having the last sentence to be" in constraint:
            # we can add a local edit suggestion here
            pattern = re.compile(r"'([^']*)'")
            matches = pattern.search(response)
            valid_last_sentences = []
            for match in matches:
                valid_last_sentences.append(match)
            response = response + " " + valid_last_sentences[0]
        return response

    @trace_op(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def create_prompt(self, prompt_template, constraint):
        """
        The function takes in a question and then add to the prompt for LLM to answer.
        Args:
            prompt_template: some guidance/hints/suggestions for LLM
            question: the question for the LLM to answer
        """
        return prompt_template.format(constraint)

    def forward(self, constraint):
        """
        question: text

        We read in a question and produces a response
        """
        user_prompt = self.create_prompt(self.prompt_template, constraint)
        response = self.call_llm(user_prompt)
        answer = self.check_and_edit_response(constraint, response)
        return answer

    def call(self, user_prompt):
        response = self.call_llm(user_prompt)
        return response

def evaluate_llm(dp, examples, nshot):
    rewards = 0
    responses = []
    reward_matrix = np.zeros((len(examples), nshot))

    for step, example in enumerate(tqdm(examples)):
        curr_try = nshot
        feedbacks = []
        curr_responses = []

        while curr_try > 0:
            GRAPH.clear()
            if len(feedbacks) == 0:
                response = dp.call(example['prompt'])
                response = response.data
            else:
                response = dp.call_llm_iterate(example['prompt'], curr_responses[-1], feedbacks[-1])
            curr_responses.append(response)
            constraint = example['constraint']
            correctness = constraint.check(response, example['targets'])
            print(response, correctness)

            if correctness:
                feedback = "The generated passage satisfies the constraint! No need to change anything."
            else:
                feedback = "The generated passage does not satisfy constraint. Modify the prompt (but keep it general for all constraints).\n\n"
                renderer = ConstraintRenderer(
                    constraint=constraint,  # Defined in step one
                    check_value=example['targets']
                )
                feedback += renderer.get_feedback(response)

            feedbacks.append(feedback)

            rewards += correctness
            reward_matrix[step, nshot - curr_try:] = correctness

            if correctness:
                break

            curr_try -= 1

        print("current accuracy:", rewards / (step + 1))

    return rewards / len(examples), responses, reward_matrix

def learn_predict(dp, optimizer, examples, nshot, save_dir):
    dp.save(f"{save_dir}/init_model.pkl")

    # save trajectory
    responses = []
    rewards = 0
    reward_matrix = np.zeros((len(examples), nshot))

    pbar = tqdm(examples)

    for step, example in enumerate(pbar):
        curr_try = nshot
        dp.load(f"{save_dir}/init_model.pkl")
        traj_data = []

        while curr_try > 0:
            GRAPH.clear()
            # This is also online optimization
            # we have the opportunity to keep changing the function with each round of interaction
            try:
                response = dp.forward(example['prompt'])
                constraint = example['constraint']
                correctness = constraint.check(response.data, example['targets'])
                if correctness:
                    feedback = "The generated passage satisfies the constraint! No need to change anything."
                else:
                    feedback = "The generated passage does not satisfy constraint. Modify the prompt (but keep it general for all constraints).\n\n"
                    renderer = ConstraintRenderer(
                        constraint=constraint,  # Defined in step one
                        check_value=example['targets']
                    )
                    feedback += renderer.get_feedback(response.data)
                no_error = True
            except TraceExecutionError as e:
                response = e.exception_node
                feedback = response.data
                correctness = False
                no_error = False

            print("Constraint:", example['prompt'])
            print("Expected example:", example['example'])
            print("Answer:", response.data)

            rewards += correctness

            # fill all the following entry as 1 or 0
            reward_matrix[step, nshot - curr_try:] = correctness
            traj_data.append({'response': response.data, 'feedback': feedback, 'correctness': correctness, 'no_error': no_error})

            # if we can handle the case, no need to optimize
            if correctness:
                # evaluate on val examples
                break

            if curr_try - 1 == 0:
                # we don't optimize for last step
                break

            optimizer.zero_feedback()
            optimizer.backward(response, feedback)

            print(f"output={response.data}, feedback={feedback}, variables=\n")  # logging
            for p in optimizer.parameters:
                print(p.name, p.data)
            optimizer.step(verbose=False)

            curr_try -= 1

        print("current accuracy:", rewards / (step + 1))
        traj_data.append({'optimizer_log': optimizer.log})
        responses.append(traj_data)

    print(f"Total accuracy: {rewards / len(examples)}")
    return dp, reward_matrix, responses, rewards / len(examples)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="FunctionOptimizerV2", help="FunctionOptimizerV2|OPRO|no_op")
    parser.add_argument("--nshot", type=int, default=4, help="FunctionOptimizerV2|OPRO|no_op")
    parser.add_argument("--save_path", type=str, default="results/collie_nshot")
    parser.add_argument("--ckpt_save_name", type=str, default="trace_collie_nshot_ckpt")
    args = parser.parse_args()

    init_data()
    all_examples = load_data(n=350)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # trainset = all_examples[:80]
    # valset = all_examples[80:100]
    test_set = all_examples[100:]

    stats = {}

    ckpt_save_name = args.ckpt_save_name
    ckpt_save_name = ckpt_save_name.replace("trace", "opro") if args.optimizer == "OPRO" else ckpt_save_name

    dp = Predict()
    if args.optimizer != 'no_op':
        if args.optimizer == "OPRO":
            params = dp.parameters()
            for p in params:
                p.trainable = False
            optimizer = OPRO([dp.prompt_template], config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
        else:
            optimizer = FunctionOptimizerV2(dp.parameters() + [dp.prompt_template],
                                        config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))
        dp, reward_matrix, responses, val_acc = learn_predict(dp, optimizer, test_set, args.nshot, ckpt_save_name)
        stats['reward_matrix'] = reward_matrix
    else:
        val_acc, responses, reward_matrix = evaluate_llm(dp, test_set, args.nshot)
        stats['reward_matrix'] = reward_matrix

    stats['val_acc'] = val_acc
    stats['val_responses'] = responses

    # stats['optimizer_log'] = optimizer.log
    # stats['train_acc'] = rewards / len(trainset)

    # stats["learned_prompt"] = dp.prompt_template.data
    # stats["extract_answer"] = dp.parameters_dict()['extract_answer'].data
    # stats["create_prompt"] = dp.parameters_dict()['create_prompt'].data

    # val_acc, responses = evaluate_dp(dp, test_set)
    # stats['val_acc'] = val_acc
    # stats['val_responses'] = responses

    import pickle

    if args.optimizer == "FunctionOptimizerV2":
        save_name = "dp_agent"
    elif args.optimizer == "no_op":
        save_name = "dp_agent_no_op"
    else:
        save_name = "dp_agent_opro"

    with open(f"{args.save_path}/{save_name}.pkl", "wb") as f:
        pickle.dump(stats, f)