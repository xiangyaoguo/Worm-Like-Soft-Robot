import pickle
import json
from mongo_atlas import get_metrics, get_config
import pandas as pd
import os
import paramiko
from scp import SCPClient
from passwords import Snellius
from sim import TrainedPolicySim
import itertools
import numpy as np

def createSCPClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return SCPClient(client.get_transport())

scp_client = None
def scp_pull(remote_path, local_path, recursive=False):
    global scp_client
    if scp_client is None:
        scp_client = createSCPClient(Snellius.server, Snellius.port, Snellius.user, Snellius.password)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # print(f"scp {Snellius.user}@{Snellius.server}:{Snellius.home_dir + remote_path} {local_path}")
    scp_client.get(Snellius.home_dir + remote_path, local_path, recursive)

class ExperimentSet:
    def __init__(self, name, run_ids, groupby, queries=[]):
        self.name = name
        self.groupby = groupby
        self._cache_path = f"temp/experiment_sets/{name}.pickle"
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        if os.path.isfile(self._cache_path):
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)
        else:
            self._cache = {}
        
        # del self._cache["run_metrics"]
        
        self.groups = self._get_cache("groups", lambda: self._init_groups(run_ids, groupby, queries))
        self.run_metrics = self._get_cache("run_metrics", lambda: self._init_run_metrics())
        self.best_group_checkpoints = self._init_best_group_checkpoints()
        for k, v in self.best_group_checkpoints.items():
            mean, std = self._evaluate_checkpoint(*v)
            # print(k, mean, std)
        
        self._init_best_group_checkpoints()
    
    def table(self, on_row, on_column):
        possible_values = {}
        for i, key in enumerate(self.groupby):
            possible_values[key] = []
            for group in self.groups:
                value = group["label"][i]
                if value not in possible_values[key]:
                    possible_values[key].append(value)
        
        rows = list(itertools.product(*[possible_values[key] for key in on_row]))
        cols = list(itertools.product(*[possible_values[key] for key in on_column]))
        scores = []
        for row in rows:
            scores.append([])
            for col in cols:
                filter = {on_row[i]: row[i] for i in range(len(on_row))} | {on_column[i]: col[i] for i in range(len(on_column))}
                key = str(tuple([filter[x] for x in self.groupby]))
                mean, std = self._evaluate_checkpoint(*self.best_group_checkpoints[key])
                scores[-1].append(mean)
        
        print("COLS:", cols)
        print("ROWS:", rows)
        for row in scores:
            print(" & ".join(["{:.2f}".format(a * 100) for a in row]), "\\\\")


        # rows = []
        # row_indices = [self.groupby.index(x) for x in on_row]
        # for group in self.groups:
        #     d = {self.groupby[i]: group["label"][i] for i in row_indices}
        #     if d not in rows:

        #     print(d)
    
    def get_speed(self, filter):
        rets = []
        other = [k for k in self.groupby if k not in filter]
        for group in self.groups:
            group_vals = [group['label'][self.groupby.index(k)] for k in filter.keys()]
            if np.all([group_vals[i] == list(filter.values())[i] for i in range(len(group_vals))]):
                rets.append({
                    "label": {o: group['label'][self.groupby.index(o)] for o in other},
                    "speeds": [self.run_metrics[run_id]['training.mean_speed']['values'] for run_id in group["ids"]]
                })
        return rets
    
    def _get_cache(self, path, generator):
        if path in self._cache:
            data = self._cache[path]
        else:
            data = generator()
            self._save_cache(path, data)
        return data

    def _save_cache(self, path, data):
        self._cache[path] = data
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._cache, f)
    
    def _init_best_group_checkpoints(self):
        data = {}
        for group in self.groups:
            key = str(group["label"])
            save_every = group["config"]["save_every"]
            best_speed = -1000000
            best_checkpoint = None
            for run_id in group["ids"]:
                speeds = self.run_metrics[run_id]['training.mean_speed']['values']
                for i in range(len(speeds) // save_every):
                    t = i * save_every
                    if speeds[t] > best_speed:
                        best_speed = speeds[t]
                        best_checkpoint = (run_id, t)
            self._retrieve_checkpoint_from_remote(*best_checkpoint)
            data[key] = best_checkpoint
        return data
    
    def _evaluate_checkpoint(self, run_id, t):
        cache = self._get_cache("checkpoint_speed", lambda: dict())
        key = f"{run_id}/{t}"
        if key in cache:
            return cache[key]
        mean, std = TrainedPolicySim(f"results/{run_id}/checkpoint_{t}.pt", num_envs=100).evaluate_speed()
        cache[key] = (mean, std)
        self._save_cache("checkpoint_speed", cache)
        return cache[key]

    def _retrieve_checkpoint_from_remote(self, run_id, t):
        path = f"results/{run_id}/checkpoint_{t}.pt"
        if os.path.isfile(path):
            return
        print(f"Retrieve {run_id}/{t} from remote")
        scp_pull(path, path)
    
    def _init_run_metrics(self):
        data = {}
        for group in self.groups:
            for run_id in group['ids']:
                data[run_id] = get_metrics(run_id)
        return data

    def _init_groups(self, run_ids, groupby, queries):
        configs = {}
        dicts = {}
        for run_id in run_ids:
            config = get_config(run_id)
            if config is not None:
                configs[run_id] = config
                config = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in config.items()}
                dicts[run_id] = config
            
        df = pd.DataFrame.from_dict(dicts, orient='index')
        for query in queries:
            df = df.query(query)
        
        print(f'df after queries has {len(df)} runs')
        groups = []
        for group_name, group_df in df.groupby(groupby):
            ids = [int(i) for i in list(group_df.index)]
            groups.append({
                "ids": ids,
                "config": configs[ids[0]],
                "label": group_name
            })
        return groups

set_shared_indep_table = ExperimentSet("shared_indep_table", range(1599,1695),
                   ['n_particles', 'observation_func', 'algorithm', 'share_parameters_policy'])
set_shared_indep_table_ring = ExperimentSet("shared_indep_table_ring", range(1695,1791),
                   ['n_particles', 'observation_func', 'algorithm', 'share_parameters_policy'])
set_tunnel = ExperimentSet("tunnel", range(2003,2067),
                   ['algorithm', 'share_parameters_policy', 'terrain_settings'])
set_stairs_crawler = ExperimentSet("stairs_crawler", range(1104,1120), ['algorithm'], queries=['scenario == "crawler"'])

if __name__ == '__main__':
    pass
    # set_tunnel.table(on_column=['terrain_settings'], on_row=['algorithm', 'share_parameters_policy'])
    # print(set_shared_indep_table_ring.best_group_checkpoints["(15, 'dth_neighbours', 'ddpg', True)"])
    # print(set_shared_indep_table_ring.best_group_checkpoints["(15, 'dth_neighbours', 'ddpg', False)"])
    # print(set_shared_indep_table_ring.best_group_checkpoints["(15, 'dth_neighbours', 'ppo', True)"])
    # print(set_shared_indep_table_ring.best_group_checkpoints["(15, 'dth_neighbours', 'ppo', False)"])
    #set_shared_indep_table_ring.table(on_column=['observation_func', 'n_particles'], on_row=['algorithm', 'share_parameters_policy'])
    print("crawler", set_shared_indep_table.best_group_checkpoints["(10, 'dth_neighbours', 'ddpg', False)"])
    print("ring", set_shared_indep_table_ring.best_group_checkpoints["(10, 'dth_neighbours', 'ddpg', False)"])