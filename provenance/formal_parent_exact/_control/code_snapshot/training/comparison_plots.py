import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mongo_atlas import get_metrics, get_config
import pandas as pd
import os
from benchmarks import get_benchmark_speed
import json

from scipy.signal import lfilter
from scipy.ndimage import gaussian_filter

def smooth(data):
    return gaussian_filter(data, sigma=3)
    # n = 11
    # return lfilter([1.0/n] * n, 1, data)



def plot_all_and_mean(ax, title, labels, datas, metric, settings):
    sorted_triplets = sorted(zip(labels, datas), key=lambda x: x[0])
    labels, datas = zip(*sorted_triplets)
    
    settings = settings[0]
    benchmark_speed = get_benchmark_speed(**settings)
    ax.axhline(benchmark_speed, ls='--', color='black', lw=1, zorder=10)

    for i, label in enumerate(labels):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean = np.mean(datas[i], axis=0)
        x = np.arange(len(mean))
        for y in datas[i]:
            ax.plot(x, smooth(y), color=color, alpha=0.2)
        ax.plot(x, smooth(mean), label=label, color=color, alpha=1)

    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()


def plot_mean_and_std(ax, title, labels, datas, metric, settings):
    """
    Plots one experiment on a given axis.

    Parameters:
        ax     -- Matplotlib axis to plot on
        title  -- Title for this subplot
        labels -- List of label names (length N)
        means  -- 2D array of shape (N, M)
        stds   -- 2D array of shape (N, M)
    """
    sorted_triplets = sorted(zip(labels, datas), key=lambda x: x[0])
    labels, datas = zip(*sorted_triplets)
    
    settings = settings[0]
    benchmark_speed = get_benchmark_speed(**settings)
    ax.axhline(benchmark_speed, ls='--', color='black', lw=1, zorder=10)

    for i, label in enumerate(labels):
        mean = np.mean(datas[i], axis=0)
        std = np.std(datas[i], axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()

def deserialise(data):
    if data is None:
        return None
    try:
        return json.loads(data)
    except:
        return data

def format_label(label, metrics):
    return ", ".join([f"{metrics[i]} = {label[i]}" for i in range(len(metrics))])

def plot_all_experiments(data, title, path, groupby, metric, sharey=True, format_title_func=format_label):
    num_experiments = len(data)
    cols = 2 if len(data) > 1 else 1
    rows = int(np.ceil(num_experiments / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharey=sharey)
    axes = axes.flatten() if num_experiments > 1 else [axes]

    for ax, (tittle, content) in zip(axes, data.items()):
        # tittle = ", ".join([f"{groupby[i]} = {tittle[i]}" for i in range(len(groupby))])
        tittle = format_title_func(tittle, groupby)
        labels = content['labels']
        datas = content['datas']
        settings = content['settings']
        plot_mean_and_std(ax, tittle, labels, datas, metric, settings)

    # Hide any unused axes
    for ax in axes[len(data):]:
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    # plt.show()
    if sharey:
        root, ext = os.path.splitext(path)
        path = f"{root}_sharey{ext}"
    plt.savefig(path)

def get_configs(run_ids, queries=[], groupby=[], comparison_metrics=['lr'], metric='training.mean_speed', format_label_func=format_label):
    dicts = {}
    for run_id in run_ids:
        config = get_config(run_id)
        # config = {'scenario': 'crawler', 'n_nodes': 13, 'algorithm': 'ddpg'}
        if config is not None:
            config = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in config.items()}
            dicts[run_id] = config
        
    df = pd.DataFrame.from_dict(dicts, orient='index')
    for query in queries:
        df = df.query(query)
    print(df)
    
    print(f'df after queries has {len(df)} runs')

    def get_group(group_name, group_df):
        labels = []
        datas = []
        settings = []
        for label, data in group_df.sort_values(comparison_metrics).groupby(comparison_metrics):
            labels.append(format_label_func(label, comparison_metrics))
            label_run_ids = list(data.index)
            print(f'label {label} has {len(data)} runs')
            metrics = [get_metrics(run_id)['training.mean_speed']['values'] for run_id in label_run_ids]
            min_time = np.min([len(a) for a in metrics])
            metrics = [a[:min_time] for a in metrics]
            metrics = np.array(metrics)
            # metrics = np.array([np.random.normal(size=100, scale=0.01) for run_id in label_run_ids])
            datas.append(np.nan_to_num(metrics))
            settings.append({
                'scenario': group_df.reset_index()['scenario'][0],
                'n_particles': group_df.reset_index()['n_particles'][0],
                'terrain_type': group_df.reset_index()['terrain_type'][0],
                'terrain_settings': deserialise(group_df.reset_index()['terrain_settings'][0])
                })
        return {'labels': labels, 'datas': datas, 'settings': settings}
    groups = {}
    if len(groupby) > 0:
        for group_name, group_df in df.groupby(groupby):
            groups[group_name] = get_group(group_name, group_df)
    else:
        groups[""] = get_group("", df)
    return groups

def make_plot(title, path, run_ids, queries=[], groupby=[], comparison_metrics=['lr'], metric='training.mean_speed', format_label_func=format_label, format_title_func=format_label):
    data = get_configs(run_ids, queries, groupby, comparison_metrics, metric, format_label_func)
    plot_all_experiments(data, title, path, groupby, metric, sharey=False, format_title_func=format_title_func)
    plot_all_experiments(data, title, path, groupby, metric, sharey=True, format_title_func=format_title_func)


def format_label_implementation_details(label, metrics):
    label0 = 'With replacement' if label[0] else 'Without replacement'
    return f"{label0}, {label[1]}"

def format_label_sharaparam_bufferreplacement(label, metrics):
    sharelabel = 'Shared policy params' if label[0] else 'Independent policy params'
    buflabel = 'With experience replay' if label[1] else 'No experience replay'
    return f"{sharelabel}, {buflabel}"

if __name__ == '__main__':
    # make_plot(
    #     title='Mean speed of Ring when training with MADDPG',
    #     path='plots/maddpg_implementation_details.png',
    #     run_ids=range(448,460),
    #     queries=['scenario == "ring"', 'algorithm == "ddpg"', 'share_parameters_policy == True', 'lr == 3e-5'],
    #     groupby=[],
    #     comparison_metrics=['buffer_sample_with_replacement', 'buffer_storage'],
    #     format_label_func=format_label_implementation_details
    # )
    # make_plot(
    #     title='Mean speed of Ring when training with MAPPO vs MADDPG',
    #     path='plots/mappo_vs_maddpg_ring.png',
    #     run_ids=range(469,517),
    #     queries=['scenario == "ring"'],
    #     groupby=['share_parameters_policy', 'buffer_sample_with_replacement'],
    #     comparison_metrics=['algorithm'],
    #     format_title_func=format_label_sharaparam_bufferreplacement
    # )
    # make_plot(
    #     title='Mean speed of Crawler when training with MAPPO vs MADDPG',
    #     path='plots/mappo_vs_maddpg_crawler.png',
    #     run_ids=range(469,517),
    #     queries=['scenario == "crawler"'],
    #     groupby=['share_parameters_policy', 'buffer_sample_with_replacement'],
    #     comparison_metrics=['algorithm'],
    #     format_title_func=format_label_sharaparam_bufferreplacement
    # )
    # make_plot(
    #     title='Mean speed of Crawler when training with MAPPO vs MADDPG',
    #     path='plots/mappo_vs_maddpg.png',
    #     run_ids=range(469,493),
    #     queries=[],
    #     groupby=[ 'scenario', 'share_parameters_policy'],
    #     comparison_metrics=['algorithm']
    # )
    # make_plot(
    #     title='Mean speed when training with MAPPO vs MADDPG and using the two neighbour angles as inputs',
    #     path='plots/mappo_vs_maddpg_obs_dth_neighbours.png',
    #     run_ids=range(524,536),
    #     queries=[],
    #     groupby=[ 'scenario'],
    #     comparison_metrics=['algorithm']
    # )
    # make_plot(
    #     title='Comparison for 500 episodes',
    #     path='plots/500_ep_comp.png',
    #     run_ids=range(595,627),
    #     queries=[],
    #     groupby=['scenario', 'buffer_sample_with_replacement'],
    #     comparison_metrics=['algorithm'],
    #     # format_title_func=format_label_sharaparam_bufferreplacement
    # )
    # make_plot(
    #     title='Comparison for 500 episodes, independent parameters',
    #     path='plots/500_ep_comp_indep.png',
    #     run_ids=range(627,659),
    #     queries=[],
    #     groupby=['scenario', 'buffer_sample_with_replacement'],
    #     comparison_metrics=['algorithm'],
    #     # format_title_func=format_label_sharaparam_bufferreplacement
    # )
    # make_plot(
    #     title='MAPPO on crawler with 13 nodes',
    #     path='plots/crawler_mappo_weight_decay.png',
    #     run_ids=range(595,708),
    #     queries=['scenario == "crawler"', 'algorithm == "ppo"', 'episodes == 500', 'buffer_sample_with_replacement == False'],
    #     groupby=['share_parameters_policy', 'share_parameters_critic'],
    #     comparison_metrics=['weight_decay']
    # )
    # make_plot(
    #     title='Shared parameters, comparison of weight decay',
    #     path='plots/mappo_maddpg_shared_weight_decay.png',
    #     run_ids=range(708,804),
    #     queries=[],
    #     groupby=['scenario', 'algorithm'],
    #     comparison_metrics=['weight_decay']
    # )
    # make_plot(
    #     title='Independent parameters, comparison of weight decay',
    #     path='plots/mappo_maddpg_indep_weight_decay.png',
    #     run_ids=range(804,900),
    #     queries=[],
    #     groupby=['scenario', 'algorithm'],
    #     comparison_metrics=['weight_decay']
    # )
    # make_plot(
    #     title='Independent parameters, pretrained with shared parameters',
    #     path='plots/mappo_maddpg_pretrained.png',
    #     run_ids=range(922,930),
    #     queries=[],
    #     groupby=[],
    #     comparison_metrics=['algorithm']
    # )
    # make_plot(
    #     title='Comparison of observation functions',
    #     path='plots/obs_func_comp.png',
    #     run_ids=range(963,1027),
    #     queries=[],
    #     groupby=['algorithm', 'share_parameters_policy'],
    #     comparison_metrics=['observation_func']
    # )
    # make_plot(
    #     title='Trying to get the crawler to roll again (no success)',
    #     path='plots/roll_again_plz.png',
    #     run_ids=range(1034,1066),
    #     queries=[],
    #     groupby=['policy_net_config'],
    #     comparison_metrics=['expl_noise']
    # )
    # make_plot(
    #     title='Moving up stairs',
    #     path='plots/stairs.png',
    #     run_ids=range(1104,1120),
    #     queries=[],
    #     groupby=['scenario'],
    #     comparison_metrics=['algorithm']
    # )
    # make_plot(
    #     title='PPO on flat ground, training for 2000 eps',
    #     path='plots/ppo_2000.png',
    #     run_ids=range(1120,1216),
    #     queries=[],
    #     groupby=['n_particles'],
    #     comparison_metrics=['share_parameters_policy']
    # )
    # make_plot(
    #     title='DDPG on flat ground, training for 2000 eps',
    #     path='plots/ddpg_2000.png',
    #     run_ids=range(1216,1280),
    #     queries=[],
    #     groupby=['n_particles'],
    #     comparison_metrics=['share_parameters_policy']
    # )
    # make_plot(
    #     title='PPO on flat ground, training for 2000 eps',
    #     path='plots/ppo_gauss_2000.png',
    #     run_ids=list(range(1120,1152)) + list(range(1345,1377)),
    #     queries=[],
    #     groupby=['n_particles'],
    #     comparison_metrics=['gaussian_activation']
    # )
    # make_plot(
    #     title='PPO numerical fixes',
    #     path='plots/ppo_numerical_fixes.png',
    #     run_ids=list(range(1120,1152)) + list(range(1382,1414))  + list(range(1417,1449)),
    #     queries=[],
    #     groupby=['n_particles'],
    #     comparison_metrics=['experiment_name']
    # )
    # make_plot(
    #     title='PPO normal scale lb',
    #     path='plots/ppo_normal_scale_lb.png',
    #     run_ids=range(1450,1490),
    #     queries=[],
    #     groupby=[],
    #     comparison_metrics=['normal_scale_lb']
    # )
    # make_plot(
    #     title='Crawler',
    #     path='plots/shared_indep_table.png',
    #     run_ids=range(1599,1695),
    #     queries=[],
    #     groupby=['n_particles', 'observation_func'],
    #     comparison_metrics=['algorithm', 'share_parameters_policy']
    # )
    # make_plot(
    #     title='Ring',
    #     path='plots/shared_indep_table_ring.png',
    #     run_ids=range(1695,1791),
    #     queries=[],
    #     groupby=['n_particles', 'observation_func'],
    #     comparison_metrics=['algorithm', 'share_parameters_policy']
    # )
    make_plot(
        title='Tunnel',
        path='plots/tunnel.png',
        run_ids=range(2003,2067),
        queries=[],
        groupby=['terrain_settings'],
        comparison_metrics=['algorithm', 'share_parameters_policy']
    )