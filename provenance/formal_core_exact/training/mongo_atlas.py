from sacred.observers import MongoObserver
from pymongo import MongoClient
from passwords import mongo_uri as uri

mongo_atlas_observer = MongoObserver(
    url=uri,
    db_name='MARLMongo'
)

client = MongoClient(uri)
db = client['MARLMongo']

def get_metrics(run_id):
    collection = db.metrics
    cursor = collection.find({'run_id': run_id})
    m = {}
    for element in cursor:
        m[element['name']] = {
            'steps': element['steps'],
            'timestamps': element['timestamps'],
            'values': element['values']
        }
    return m

DEFAULT_CONFIG = {'weight_decay': 0.0, 'gaussian_activation': False}
def get_config(run_id):
    exp_info = db.runs.find_one({"_id": run_id})
    if exp_info is None:
        return None
    cfg = exp_info['config']
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = value
    cfg['experiment_name'] = exp_info['experiment']['name']
    return cfg