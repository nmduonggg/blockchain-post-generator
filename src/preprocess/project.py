import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import json
from pymongo import MongoClient

import preprocess.utils as utils

# Convert to datetime object
def to_datetime(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')

client = MongoClient("mongodb://dsReader:ds_reader_ndFwBkv3LsZYjtUS@178.128.85.210:27017,104.248.148.66:27017,103.253.146.224:27017")

# print(client.list_database_names())

db = client['knowledge_graph']
# print(db.list_collection_names())
project_collection = db['projects']

def query_trans_in_block_hash(block_hash, ub, lb):
    query = {
        'timestamp': {'$lte': ub, '$gte': lb},
        'block_hash': block_hash,
    }
    return query
    

def input():

    now = datetime(2024, 12, 8, 0, 0, 0, tzinfo=pytz.utc)
    month_ago = now - timedelta(days=1)
    month_ago = month_ago.timestamp()

    now = now.timestamp()

    query = {
        'timestamp': {'$lte': now,
                    '$gte': month_ago},
    }
    
    # block querying
    projects = project_collection.find()
    list_projects = []
    
    categories = {}
    idx = 0
    n_samples = 50000
    for block in tqdm(projects, total=n_samples):
        idx += 1
        if 'category' not in block: continue
        
        list_projects.append(block)
            
        category = block['category']
        cate_item = categories.get(category, {})
        
        current_keys = cate_item.get('key', set())
        cate_data = {
            'keys': current_keys.union(set(block.keys())) if len(current_keys)==0 else \
                current_keys.intersection(set(block.keys())),
            'num_instances': cate_item.get('num_instances', 0) + 1
        }
        categories[category] = cate_data
        
    categories ={k: v for k, v in sorted(categories.items(), key=lambda x: x[1]['num_instances'], reverse=True)}
    
    top_k = 5
    for cnt, (k, v) in enumerate(categories.items()):
        print(k, v['num_instances'])
        print(v['keys'])
        print("---"*5 + '\n')
        if cnt==top_k: break
        
    with open('./list_projects.json', 'w') as f:
        json.dump(list_projects, f, indent=4)
    

def demo_input():
    with open('./list_projects.json', 'r') as f:
        data = json.load(f)
    return data

documents = demo_input()

utils.projects_cnt_top(documents)

# blocks, transactions = input()
# input()



