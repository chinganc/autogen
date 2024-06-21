from autogen.trace import bundle, node, Module
from autogen.trace.nodes import ExceptionNode
from autogen.trace.bundle import TraceExecutionError
from autogen.trace.optimizers import FunctionOptimizerV2, FunctionOptimizerV2Memory, OPRO
from autogen.trace.nodes import GRAPH
from battleship import BattleshipBoard
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import pickle
import os
from datetime import datetime


def user_fb_for_placing_shot(board, coords):
    # this is already a multi-step cumulative reward problem
    # obs, reward, terminal, feedback
    try:
        reward = board.check_shot(coords[0], coords[1])
        new_map = board.get_shots()
        terminal = board.check_terminate()
        return new_map, reward, terminal, f"Got {int(reward)} reward."
    except Exception as e:
        return board.get_shots(), 0, False, str(e)


def rollout(policy, board_width, board_height, num_each_type, exclude_ships, horizon):
    board = BattleshipBoard(board_width, board_height, num_each_type=num_each_type, exclude_ships=exclude_ships)
    rewards = []
    obs = board.get_shots()  # init observation
    for i in range(horizon):
        output = policy(obs)
        obs, reward, terminal, feedback = user_fb_for_placing_shot(board, output)  # not traced
        if terminal:
            break
        rewards.append(reward)
    rewards = np.array(rewards)
    return rewards


def eval_policy(policy, args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    scores = []
    for _ in range(args.n_eval_episodes):
        # board_width = np.random.randint(args.board_size, int(2 * args.board_size))
        # board_height = np.random.randint(args.board_size, int(2 * args.board_size))
        # horizon = board_width * board_height
        board_width = args.board_size
        board_height = args.board_size
        horizon = (board_width * board_height) // 2

        policy.init(board_width, board_height)
        rewards = rollout(policy, board_width, board_height, args.num_each_type, args.exclude_ships, horizon)
        scores.append(rewards.mean())
    scores = np.array(scores)
    print(f"Scores: {scores.mean()} ({scores.std()})")
    return scores


def run(args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init variable
    # Given a map, select a valid coordinate to earn reward.

    # a wrapper for the eval api

    class Policy(Module):
        def init(self, width, height):
            pass

        def __call__(self, map):
            return self.select_coordinate(map).data

        @bundle(trainable=True)
        def select_coordinate(self, map):
            """
            Given a map, select a target coordinate in a Battleship game. In map, O denotes misses, X denotes successes, and . denotes unknown positions.
            """
            return [0, 0]

    class Policy2(Module):
        def init(self, width, height):
            pass

        def __call__(self, map):
            return self.select_coordinate(map).data

        def select_coordinate(self, map):
            plan = self.reason(map)
            output = self.act(map, plan)
            return output

        @bundle(trainable=True)
        def act(self, map, plan):
            """
            Given a map, select a target coordinate in a game.
            X denotes hits, O denotes misses, and . denotes unknown positions.
            """
            return

        @bundle(trainable=True)
        def reason(self, map):
            """
            Given a map, analyze the board in a game.
            X denotes hits, O denotes misses, and . denotes unknown positions.
            """
            return

    policy = Policy2()

    def reset_board():
        # square board
        board = BattleshipBoard(
            args.board_size, args.board_size, num_each_type=args.num_each_type, exclude_ships=args.exclude_ships
        )
        if args.visualize:
            print("Ground State Board")
            board.visualize_board()
        return board

    optimizer = args.opt_cls(policy.parameters())

    writer = SummaryWriter(args.logdir)
    log = defaultdict(list)

    # init eval
    returns = eval_policy(policy, args)
    log["returns"].append(returns)
    writer.add_scalar("Evaluation/mean score", returns.mean(), 0)

    feedback = ""
    rewards = []
    # This is an online optimization problem. we have the opportunity to
    # keep changing the function with each round of interaction
    board = reset_board()
    obs = node(board.get_shots())  # init observation
    i = 0
    while i < args.max_calls:
        GRAPH.clear()

        print("Iteration", i)
        try:
            output = policy.select_coordinate(obs)
            obs, reward, terminal, feedback = user_fb_for_placing_shot(board, output.data)  # not traced
        except TraceExecutionError as e:  # this is a retry
            output = e.exception_node
            feedback = output.data
            reward, terminal = 0, False

        if terminal:
            board = reset_board()
            obs = node(board.get_shots())  # init observation

        # Update
        optimizer.zero_feedback()
        optimizer.backward(output, feedback)
        optimizer.step(verbose=True)

        # Logging
        if not isinstance(output, ExceptionNode):
            rewards.append(reward)
            if args.visualize:
                print("Obs:")
                board.visualize_shots()
                print(f"output={output.data}, feedback={feedback}, variables=\n")  # logging
                for p in optimizer.parameters:
                    print(p.name, p.data)
            try:
                returns = eval_policy(policy, args)
                log["returns"].append(returns)
                log["instant reward"].append(reward)
                log['parameters'] = {'reason': '', 'act': ''}
                log['parameters']['reason'] = policy.parameters_dict['reason'].data
                log['parameters']['act'] = policy.parameters_dict['act'].data

                writer.add_scalar("Evaluation/mean score", returns.mean(), i + 1)  # i+1 to account for the initial log
                writer.add_scalar("Training/instant reward", reward, i)

                writer.flush()
            except Exception:
                pass

        i += 1
        policy.save(f"{args.logdir}/policy_epoch{i}.pkl")
        pickle.dump(log, open(f"{args.logdir}/log.pkl", "wb"))

    rewards = np.array(rewards)
    print(f"Cumulative rewards: {rewards.sum()}")
    print(f"Average rewards: {rewards.mean()}")
    return rewards


class BasicEnumerator:
    def init(self, width, height):
        self.width = width
        self.height = height
        self.i = 0
        self.j = 0

    def __call__(self, obs):
        if self.i == self.width:
            self.i = 0
            self.j += 1
        if self.j == self.height:
            raise StopIteration
        self.i += 1
        return [self.i, self.j]


class RandomPolicy(BasicEnumerator):
    def __call__(self, obs):
        return [np.random.randint(self.width), np.random.randint(self.height)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, default=5)
    parser.add_argument("--num_each_type", type=int, default=1)
    parser.add_argument("--exclude_ships", nargs="+", type=str, default=("C"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_calls", type=int, default=10)
    parser.add_argument("--memory_size", type=int, default=0)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="battleship_results3")
    parser.add_argument("--optimizer", type=str, default="FunctionOptimizerV2Memory")
    args = parser.parse_args()

    # Log
    args.logdir += (
        f"/{args.optimizer}_mem{args.memory_size}/seed_{args.seed}_size{args.board_size}_num{args.num_each_type}"
    )
    args.logdir += datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.logdir, exist_ok=True)

    pickle.dump(args, open(f"{args.logdir}/args.pkl", "wb"))

    print("Enumeration")
    scores = eval_policy(BasicEnumerator(), args)
    pickle.dump(scores, open(f"{args.logdir}/enumeration_scores.pkl", "wb"))
    print("Random")
    scores = eval_policy(RandomPolicy(), args)
    pickle.dump(scores, open(f"{args.logdir}/random_policy_scores.pkl", "wb"))

    if args.optimizer == "FunctionOptimizerV2Memory":
        args.opt_cls = lambda *_args, **_kwargs: FunctionOptimizerV2Memory(
            *_args, **_kwargs, memory_size=args.memory_size
        )
    elif args.optimizer == "OPRO":
        args.opt_cls = OPRO
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    run(args)
