
"""
functions:
    process_all_datasets
    get_datasets
    shorten_datasets
    align_datasets
    process_steps
    plot
"""

import seaborn as sns
from seaborn.categorical import boxplot; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import yaml

DIV_LINE_WIDTH = 50

STYLE = ['--', '-.', ':', '-']
LINEWIDTH = [1.5, 1.5, 2, 1.5]
LINECOLOR = 'm'

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def process_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    # print(all_logdirs)
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)),         "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                config_path = os.path.join(root, 'config.yml')
                # print(config_path)
                if os.path.isfile(config_path):
                    f = open(config_path)
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    if 'exp_name' in config:
                        exp_name = config['exp_name']
                else:
                    print("Configuration file config.json and config.yml is not found in %s"%(root))
                # print('No file named config.json')

            if "rce" in exp_name:
                exp_name = "MPC-RCE(ours)"
            exp_name = "MPC-CEM" if "cem" in exp_name else exp_name
            exp_name = "MPC-random" if "random" in exp_name else exp_name

            condition1 = condition or exp_name or 'exp'  # differentiate by method
            condition2 = condition1 + '-' + str(exp_idx)  # differentiate by seed
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            pro_path = os.path.join(root, 'progress.txt')
            print("reading data from %s" % (pro_path))

            try:
                exp_data = pd.read_table(pro_path)
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue

            '''
            **********************************************
            process the data and --replace-- the original file
            **********************************************
            '''
            # process_and_replace_data(exp_data, pro_path)

            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Method', condition1)
            exp_data.insert(len(exp_data.columns), 'BySeed', condition2)

            datasets.append(exp_data)
    return datasets


def shorten_datasets(datasets, cut, condition):
    """
    cut 2 - no action
        1 - cut catergorized by condition
        0 - cut to global shortest
    """
    if cut > 1:
        return datasets

    column_length = dict()
    shortest_length = None

    for dataset in datasets:
        method = dataset[condition][0]  # method name of the exp
        length = len(dataset)  # length of column
        if method not in column_length:  # add to dict
            column_length[method] = length
        if length < column_length[method]:  # update dict with lower length
            column_length[method] = length
        if shortest_length is None:
            shortest_length = length
        if length < shortest_length:
            shortest_length = length

    # print(column_length)

    new_datasets = []
    for dataset in datasets:
        # dataset.drop([column_length[dataset['Method'][0]]:end])
        if cut == 0:
            dataset = dataset[:shortest_length]
        elif cut == 1:
            dataset = dataset[:column_length[dataset[condition][0]]]

        new_datasets.append(dataset)

    return new_datasets


def align_datasets(datasets, x_label, condition):
    """
        align the datasets' x_labels, grouped by condition
    """
    x_align = dict()
    size_align = dict()

    for dataset in datasets:
        cond_name = dataset[condition][0]
        x_dataset = dataset[x_label]
        last = x_dataset[len(dataset)-1]
        if cond_name not in x_align:
            x_align[cond_name] = x_dataset
            size_align[cond_name] = last
        if last < size_align[cond_name]:
            x_align[cond_name] = x_dataset
            size_align[cond_name] = last

    for dataset in datasets:
        dataset[x_label] = x_align[dataset[condition][0]]

    return datasets


def process_steps(datasets):
    for data in datasets:
        data['Steps'] -= data['Steps'].values[0]
        data['Steps'] += 1
    return datasets


def process_rewards_cost(datasets):
    for dataset in datasets:
        names = list(dataset.columns)
        if 'CumulativeCost' not in names:
            data = []
            data.append(dataset['Cost'].values[0])
            for i in range(1, dataset.shape[0]):
                data.append(data[-1] + dataset['Cost'].values[i])
            dataset['CumulativeCost'] = data
        if 'MaximumReward' not in names:
            data = []
            max_reward = 0.0
            for i in range(dataset.shape[0]):
                if dataset['Reward'].values[i] > max_reward:
                    max_reward = dataset['Reward'].values[i]
                data.append(max_reward)
            dataset['MaximumReward'] = data
    return datasets


def rename(datasets):
    datasets = datasets.replace(
                    {"Method": {
                        "ppo_lagrangian": "ppo_lag",
                        "trpo_lagrangian": "trpo_lag"
                    }})
    return datasets


def process_data(logdir, x_label='TotalEnvInteracts', y_label='EpRet', cut=0, condition='Method', smooth=50):
    # print(logdir)
    datasets = process_all_datasets([logdir])
    # array of DataFrame 's
    # each column is a Series
    datasets = shorten_datasets(datasets, cut, condition)
    datasets = align_datasets(datasets, x_label, condition)

    for df in datasets:
        df.rename(columns={"EpCost": "Cost", "worker/EpCost": "Cost", "worker/EpRet": "Reward", "EpRet": "Reward", "AverageEpCost": "Cost", "AverageEpRet": "Reward",
                           "TotalEnvInteracts": "Steps"}, inplace=True)

    x_label = "Steps" if x_label == "TotalEnvInteracts" else x_label
    y_label = "Cost" if y_label == "EpCost" or y_label == "AverageEpCost" else y_label
    y_label = "Reward" if y_label == "EpRet" or y_label == "AverageEpRet" else y_label

    datasets = process_steps(datasets)
    datasets = process_rewards_cost(datasets)
    return datasets


def plot_data(data_in, axe, x_label, y_label, condition, smooth, hue_order=None, cost_limit=1):
    """
        plot datasets to figure
    """
    datasets = data_in.copy()
    # smooth
    if smooth > 1:
        # smooth done by taking nearby averages
        smooth = min(smooth, len(datasets[0]))
        y = np.ones(smooth)
        for dataset in datasets:
            x = np.asarray(dataset[y_label])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            dataset[y_label] = smoothed_x

    # merge multiple datasets into one for 'lineplot'
    # each dataset differentiated by condition
    datasets = pd.concat(datasets, ignore_index=True)
    datasets = rename(datasets)
    # Graphics
    # plot data curves
    sns.set(style="darkgrid", font_scale=1.5)
    lp = sns.lineplot(ax=axe, data=datasets, x=x_label, y=y_label, hue=condition, ci='sd', hue_order=hue_order)
    # lp = sns.lineplot(data=datasets, x=x_label, y=y_label, hue=condition, ci='sd')
    # sns.tsplot(data=datasets, time=x_label, value=y_label, unit="Unit", condition=condition, ci='sd')

    # plot straight lines
    if y_label == "Cost":
        axe.axhline(y=cost_limit, linestyle="--", label="cost limit", linewidth=2.5)
    axe.legend("", frameon=False)
    axe.set_xscale('log')
    axe.set_xlim([1e3, 0.8e6])


def plot_cost_reward(datasets, axe, condition, hue_order=None, warmup_cost=3500):
    methods = dict()
    for i, dataset in enumerate(datasets):
        method = dataset['Method'].values[0]
        if method not in methods.keys():
            methods[method] = [i]
        else:
            methods[method].append(i)
    new_datasets = []
    for method, idxes in methods.items():
        costs = set()
        for i in range(datasets[0].shape[0]):
            for j in idxes:
                costs.add(datasets[j]['CumulativeCost'].values[i])
        costs = np.array(sorted(list(costs)))

        cumulative_costs = np.zeros((costs.shape[0], len(idxes)))
        cumulative_rewards = np.zeros((costs.shape[0], len(idxes)))
        for i in range(len(idxes)):
            cumulative_costs[:, i] = np.array(costs)
        for i in range(len(idxes)):
            idx = idxes[i]
            for j in range(costs.shape[0]):
                cost = costs[j]
                mask = datasets[idx]['CumulativeCost'] == cost
                if (np.sum(mask) == 0):
                    continue
                reward = np.max(datasets[idx]['MaximumReward'].values[mask])
                cumulative_rewards[j, i] = reward
        for i in range(1, cumulative_rewards.shape[0]):
            for j in range(cumulative_rewards.shape[1]):
                if cumulative_rewards[i, j] < cumulative_rewards[i-1, j]:
                    cumulative_rewards[i, j] = cumulative_rewards[i-1, j]
        if method != 'MPC-RCE':
            costs -= warmup_cost
        for i in range(len(idxes)):
            tmp_data = pd.DataFrame()
            cc_mask = costs >= 0
            if method == 'MPC-RCE':
                tmp_data['CumulativeCost'] = cumulative_costs[cc_mask, i]
            else:
                if np.sum(cc_mask) == 0:
                    tmp_data['CumulativeCost'] = cumulative_costs[:, i]
                else:
                    tmp_data['CumulativeCost'] = cumulative_costs[cc_mask, i] - warmup_cost
            if np.sum(cc_mask) == 0:
                tmp_data['MaximumReward'] = cumulative_rewards[:, i]
                tmp_data['Method'] = [method] * len(cc_mask)
            else:
                tmp_data['MaximumReward'] = cumulative_rewards[cc_mask, i]
                tmp_data['Method'] = [method] * np.sum(cc_mask)
            new_datasets.append(tmp_data)
    new_datasets = pd.concat(new_datasets, ignore_index=True)
    new_datasets = rename(new_datasets)

    lp = sns.lineplot(ax=axe, data=new_datasets, x='CumulativeCost', y='MaximumReward', hue=condition, ci='sd', hue_order=hue_order)
    axe.legend("", frameon=False)
    axe.set_xscale('log')
    # axe.set_xlim([0, 1e6])


def box_plot(datasets, ax, env_name, cost_limit, smooth=30):
    if smooth > 1:
        # smooth done by taking nearby averages
        smooth = min(smooth, len(datasets[0]))
        y = np.ones(smooth)
        for dataset in datasets:
            x = np.asarray(dataset['Cost'])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            dataset['Cost'] = smoothed_x
    tmp_datasets = datasets.copy()
    for dataset in tmp_datasets:
        total_steps = dataset['Steps'].values[0]
        dataset = dataset[dataset['Steps'] > total_steps * 0.9]
    new_datasets = []
    for j, method in enumerate(hue_order):
        for d in tmp_datasets:
            if d['Method'].values[0] == method:
                new_datasets.append(d)
    new_datasets = pd.concat(new_datasets, ignore_index=True)

    ax.set_title(env_name, fontsize=24)
    sns.set(style="darkgrid", font_scale=1.5)
    lp = sns.boxplot(x='Method', y="Cost", data=new_datasets, showfliers=False)
    ax.axhline(y=cost_limit, linestyle="--", label="cost limit", linewidth=2.5)
    ax.legend("", frameon=False)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set(xlabel=None)
    # ax.set_xlabel('Method', fontsize=18)
    ax.set_ylabel('Cost', fontsize=18)


if __name__ == "__main__":

    # boxplot for model-based
    logdirs = ['data/pg1', 'data/pg2', 'data/cg1', 'data/cg2']
    datasets = []
    for logdir in logdirs:
        data = process_data(logdir)
        datasets.append(data)
    hue_order = ['MPC-RCE(ours)', 'MPC-CEM', 'MPC-random']
    env_names = ['Point-Goal1', 'Point-Goal2', 'Car-Goal1', 'Car-Goal2']
    cost_limits = [1, 2, 1, 2]
    rs = [0, 0, 1, 1]
    cs = [0, 1, 0, 1]

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    for i, data in enumerate(datasets):
        ax = fig.add_subplot(gs[rs[i], cs[i]])
        box_plot(data, ax, env_names[i], cost_limits[i])
    plt.legend(loc='best').set_draggable(True)
    plt.legend(bbox_to_anchor=(-1.4, 3.7), loc='upper center', ncol=9, handlelength=1, borderaxespad=0., prop={'size': 18})

    # boxplot for model-free
    hue_order = ['MPC-RCE(ours)', 'cpo', 'ppo', 'trpo', 'ppo_lag', 'trpo_lag', 'ddpg_lag', 'sac_lag']
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.3)
    for i, data in enumerate(datasets):
        ax = fig.add_subplot(gs[rs[i], cs[i]])
        box_plot(data, ax, env_names[i], cost_limits[i])
    plt.legend(loc='best').set_draggable(True)
    plt.legend(bbox_to_anchor=(-1.4, 3.7), loc='upper center', ncol=9, handlelength=1, borderaxespad=0., prop={'size': 18})

    # overview results
    datasets = []
    for logdir in logdirs:
        data = process_data(logdir)
        datasets.append(data)

    hue_order = ['MPC-RCE(ours)', 'cpo', 'ppo', 'trpo', 'ppo_lag', 'trpo_lag', 'ddpg_lag', 'sac_lag']
    columns = len(datasets)
    # fig = plt.figure(figsize=(20, 11))
    fig = plt.figure()
    # gs = fig.add_gridspec(3, columns, hspace=0.1)
    widths = [10]
    heights = [14, 7]
    gs = fig.add_gridspec(2, 1, width_ratios=widths, height_ratios=heights)
    gs0 = gs[0].subgridspec(2, 4, hspace=0.05)
    gs1 = gs[1].subgridspec(1, 4)
    # axes = gs.subplots(sharex='col')
    env_names = ['Point-Goal1', 'Point-Goal2', 'Car-Goal1', 'Car-Goal2']
    cost_limits = [1, 2, 1, 2]
    values = ['Reward', 'Cost']
    for i, data in enumerate(datasets):
        # axes[0, i].set_title(env_names[i], fontsize=18)
        for j, value in enumerate(values):
            ax = fig.add_subplot(gs0[j, i])
            if j == 0:
                ax.get_xaxis().set_visible(False)
                ax.set_title(env_names[i], fontsize=24)
            plot_data(data, axe=ax, x_label='Steps', y_label=value, condition='Method', smooth=50, hue_order=hue_order, cost_limit=cost_limits[i])
            # change y tick text size
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_ylabel(value, fontsize=18)
            # hide y labels
            if i != 0:
                ax.set(ylabel=None)
            if j == 1:
                ax.set_xlabel('Steps', fontsize=18)

        ax = fig.add_subplot(gs1[0, i])
        plot_cost_reward(data, axe=ax, condition='Method', hue_order=hue_order, warmup_cost=3500)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel('Cumulative Cost', fontsize=18)
        ax.set_ylabel('Maximum Reward', fontsize=18)
        if i != 0:
            ax.set(ylabel=None)

    plt.legend(loc='best').set_draggable(True)
    plt.legend(bbox_to_anchor=(-1.4, 3.7), loc='upper center', ncol=9, handlelength=1, borderaxespad=0., prop={'size': 18})
    plt.show()
