import matplotlib.pyplot as plt
import sys, os
import pickle
import numpy as np
import scipy

import matplotlib
def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 18,
              'axes.labelsize': 18,
              'font.size': 18,
              'legend.fontsize': 18,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)

latexify()

def backfill(regret, maxlen):
    filled_regret = []
    for i in range(maxlen):
        if i < len(regret):
            filled_regret.append(regret[i])
        else:
            filled_regret.append(regret[-1])
    return filled_regret
def load_optimizer_results(name):
    optimizer_results = []
    # loop through all directories under a directory
    # use os.listdir(f"battleship_results/{name}/")
    max_len = 21
    for folder in os.listdir(f"battleship_results/{name}"):
        # loop through all files in the directory
        file = "log.pkl"
        with open(f"battleship_results/{name}/{folder}/{file}", "rb") as f:
            log = pickle.load(f)
            step_eval_mean = [np.mean(r) for r in log['returns']]
            optimizer_results.append(step_eval_mean)
            max_len = max(max_len, len(step_eval_mean))

    # backfill
    filled_rewards = [backfill(regret, max_len) for regret in optimizer_results]
    print("max length:", max_len)
    return np.array(filled_rewards)

def load_optimizer_results(name):
    optimizer_results = []
    random_policy_results = []
    enumeration_policy_results = []

    # loop through all directories under a directory
    # use os.listdir(f"battleship_results/{name}/")
    # max_len = 21
    max_len = 10
    for folder in os.listdir(f"battleship_results3/{name}"):
        # loop through all files in the directory
        file = "log.pkl"
        with open(f"battleship_results3/{name}/{folder}/{file}", "rb") as f:
            log = pickle.load(f)
            # import pdb; pdb.set_trace()
            step_eval_mean = [np.mean(np.array(r) * 12 / 12) for r in log['returns']]  # per training step, we eval
            optimizer_results.append(step_eval_mean)
            max_len = max(max_len, len(step_eval_mean))

        scores = pickle.load(open(f"battleship_results3/{name}/{folder}/random_policy_scores.pkl", "rb"))
        scores = (np.array(scores)) #.mean() # * 25 / 12
        random_policy_results.append(scores)
        scores = pickle.load(open(f"battleship_results3/{name}/{folder}/enumeration_scores.pkl", "rb"))
        scores = (np.array(scores))#.mean() #
        enumeration_policy_results.append(scores)

    # backfill
    filled_rewards = [backfill(regret, max_len) for regret in optimizer_results]
    print("max length:", max_len)
    return np.array(filled_rewards), (np.array(random_policy_results), np.array(enumeration_policy_results))

def plot_optimizer_results(optimizer_results, baseline_results):

    # max_reward = 0.45400

    plt.figure(figsize=(15, 4))  # Long horizontal figure
    horizon_cutoff=8

    plt.grid(True)  # Enable grid
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top border
    ax.spines['right'].set_visible(False)  # Remove right border

    for name, filled_rewards in optimizer_results.items():

        # filled_rewards /= max_reward

        reward_means = np.mean(filled_rewards, axis=0)[:horizon_cutoff]* 100
        reward_sems = scipy.stats.sem(filled_rewards, axis=0)[:horizon_cutoff]* 100
        if name == 'Trace':
            name = "Trace (Ours)"

        plt.plot(reward_means, label=name, marker='o', markersize=8, linewidth=3)
        plt.fill_between(range(len(reward_means)), reward_means - reward_sems, reward_means + reward_sems, alpha=0.4)

    # base policy we just do 2 lines
    cs = ['#2aa02b', '#d72a2c']
    i = 0
    for name, filled_rewards in baseline_results.items():
        reward = filled_rewards[0].mean() * 100
        reward_sems = scipy.stats.sem(filled_rewards[0]) * 100
        # plt.axhline(reward, 0, len(reward_means), label=name, linestyle="--", c=cs[i])  # , alpha=0.5
        plt.hlines(reward, 0, len(reward_means)-1, label=name, linestyle="--", colors=cs[i])  # , alpha=0.5
        plt.fill_between(range(len(reward_means)), reward - reward_sems, reward + reward_sems, alpha=0.2, color=cs[i])
        i += 1

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    plt.tight_layout()
    plt.xlim(1, len(reward_means)-1+0.2)
    plt.xlabel("Training Iterations")
    plt.ylabel("% of Ships Hit")
    plt.xticks(range(len(reward_means)), np.array(range(len(reward_means))))
    plt.savefig('battleship_cumulative.png', dpi=120, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':

    results = {}
    name = "FunctionOptimizerV2Memory_mem0"
    optimizer_results, base_results = load_optimizer_results(name)
    base_policy_results = {}
    base_policy_results['Enumeration'] = base_results[1]

    print(optimizer_results.max())
    results["Trace"] = optimizer_results
    name = "OPRO_mem0"
    optimizer_results, base_results = load_optimizer_results(name)
    results["OPRO"] = optimizer_results  # 10 seeds, 20 evals

    print(results)

    print(base_results[1])
    print(base_results[1].shape)  # 10 seeds, 20 evals

    # base_policy_results['Random'] = base_results[0]

    plot_optimizer_results(results, base_policy_results)