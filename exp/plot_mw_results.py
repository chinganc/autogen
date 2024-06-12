import os
import pickle
import re
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


pattern = ""


def dict_to_list(log):
    length = max([k for k in log.keys()])
    results = [np.nan] * (length + 1)

    for k, v in log.items():
        results[k] = v
    return results


def load_data(exp_dir):
    results = list()
    for d in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, d)) and re.search(pattern, d):
            sub_exp_dir = os.path.join(exp_dir, d)
            files = os.listdir(sub_exp_dir)
            event_file = [f for f in files if "events" in f]
            if len(event_file) == 0:
                continue
            else:
                assert len(event_file) == 1
                event_file = event_file[0]
            # process the event file
            data = defaultdict(dict)
            for e in summary_iterator(os.path.join(sub_exp_dir, event_file)):
                for v in e.summary.value:
                    data[v.tag][e.step] = v.simple_value
            for k, v in data.items():
                data[k] = dict_to_list(v)

            seed = re.search(r"seed_(\d+)_", d).group(1)

            result = dict(
                seed=seed,
                data=data,
                env_name=env_name,
                sub_exp_dir=sub_exp_dir,
            )
            results.append(result)
    print(f"Found {len(results)} results in {exp_dir}")
    return results


# def plot(results):
#     # Plot
#     # plot_dir = os.path.join(exp_dir, 'plots')
#     plot_dir = os.path.join(exp_dir, 'plots')

#     os.makedirs(plot_dir, exist_ok=True)
#     for k in results[0]['data'].keys():
#         stat = []
#         for r in results:
#             stat.append(np.array(r['data'][k]))
#         max_len = max([len(s) for s in stat])
#         # stat = [s[:min_len] for s in stat]
#         # fill the shorter ones with nan
#         for i in range(len(stat)):
#             if len(stat[i]) < max_len:
#                 stat[i] = np.concatenate([stat[i], [np.nan] * (max_len - len(stat[i]))])
#         # plot stat with confidence interval
#         stat = np.array(stat)
#         mean = np.nanmean(stat, axis=0)
#         std = np.nanstd(stat, axis=0)
#         # compute the non-nan values along axis 1
#         non_nan = np.count_nonzero(~np.isnan(stat), axis=0)
#         ste = std / np.sqrt(non_nan)
#         # plot
#         plt.plot(mean, label=k)
#         plt.fill_between(range(max_len), mean-ste, mean+ste, alpha=0.2)
#         # save plot
#         plt.legend()
#         file_name = k.replace('/', '_')
#         plt.savefig(f'{plot_dir}/{file_name}.png')
#         plt.show()
#         plt.close()


def compute_stat(results, key):
    stat = []
    for r in results:
        try:
            data = np.array(r["data"][key])
            len(data)
            stat.append(data)
        except TypeError:
            pass
        # stat.append(np.array(r['data'][key]))
    max_len = max([len(s) for s in stat])

    # fill the shorter ones with nan
    for i in range(len(stat)):
        if len(stat[i]) < max_len:
            stat[i] = np.concatenate([stat[i], [np.nan] * (max_len - len(stat[i]))])

    stat = np.array(stat)

    if "Evaluation/" in key:
        # XXX This is because the logged valued are sum not mean over 10 episodes
        stat /= 10

    mean = np.nanmean(stat, axis=0)
    std = np.nanstd(stat, axis=0)
    # compute the non-nan values along axis 1
    non_nan = np.count_nonzero(~np.isnan(stat), axis=0)
    ste = std / np.sqrt(non_nan)

    return mean, ste


def plot_comparison(data, plot_dir, legend=True):
    # Plot
    # plot_dir = os.path.join(exp_dir, 'plots')

    os.makedirs(plot_dir, exist_ok=True)

    font1 = {"family": "serif", "size": 15}  # legend
    font2 = {"family": "serif", "size": 20}  # labels

    keys = [v for v in data.values()][0][0]["data"].keys()
    for k in keys:
        for name, results in data.items():
            # plot stat with confidence interval
            mean, ste = compute_stat(results, k)
            plt.plot(mean, label=name)
            plt.fill_between(range(len(mean)), mean - ste, mean + ste, alpha=0.2)

        if "return" in k.lower():
            plt.ylabel("Return", fontdict=font2)
        elif "success" in k.lower():
            plt.ylabel("Success", fontdict=font2)

        plt.xlabel("Iterations", fontdict=font2)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # save plot
        if legend:
            plt.legend(prop=font1)
        file_name = k.replace("/", "_")
        plt.savefig(f"{plot_dir}/{file_name}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    rootdir = "final_exps/mw_new_exps"
    # rootdir = 'exp_results/gpt4v'
    # env_name = 'llf-metaworld-push-v2'
    # env_name = 'llf-metaworld-pick-place-v2'
    # env_name = "llf-metaworld-reach-v2"

    env_names = ["llf-metaworld-push-v2", "llf-metaworld-pick-place-v2", "llf-metaworld-reach-v2"]
    for env_name in env_names:
        data = {}

        logdir = os.path.join(rootdir, "opro")
        exp_dir = os.path.join(logdir, env_name)
        data["OPRO"] = load_data(exp_dir)

        logdir = os.path.join(rootdir, "trace_memory10")
        exp_dir = os.path.join(logdir, env_name)
        data["Trace"] = load_data(exp_dir)

        logdir = os.path.join(rootdir, "trace")
        exp_dir = os.path.join(logdir, env_name)
        data["Trace NoMem"] = load_data(exp_dir)

        logdir = os.path.join(rootdir, "baseline")
        exp_dir = os.path.join(logdir, env_name)
        data["Trace Masked"] = load_data(exp_dir)

        plot_dir = os.path.join(rootdir, "plots", env_name)

        plot_comparison(data, plot_dir, legend="reach" in env_name)
