import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import json
from pymongo import MongoClient

import preprocess.utils as utils

KEY = "mongodb://dsReader:ds_reader_ndFwBkv3LsZYjtUS@178.128.85.210:27017,104.248.148.66:27017,103.253.146.224:27017"

class BlocksCrawler():
    def __init__(self, type, n_samples):
        assert(type in ['blockchain_etl', 'ethereum_blockchain_etl']), 'Invalid block type'
        self.client = MongoClient(KEY)
        self.db = self.client[type]
        
        self.block_collection = self.db['blocks']
        self.n_samples = n_samples
    
    def query_from_time(self, given_time, prev_days, save_fn):
        ub = given_time
        lb = given_time - timedelta(days=prev_days)
        ub, lb = ub.timestamp(), lb.timestamp()
        
        query = {
            'timestamp': {'$lte': ub, '$gte': lb},
        }
        
        cnt = 0
        list_blocks = []
        blocks = self.block_collection.find(query)
        
        for block in tqdm(blocks, total=self.n_samples):
            list_blocks.append(block)
            if cnt == self.n_samples: break
            cnt += 1
            
        self.save_to_json(list_blocks, save_fn)
    
    def save_to_json(self, data, fn):
        with open(fn, 'w') as f:
            json.dump(data, f, indent=4)
            
class ProjectsCrawler():
    def __init__(self, n_samples):
        self.client = MongoClient(KEY)
        self.db = self.client['knowledge_graph']
        self.projection_collection = self.db['projects']

        self.n_samples = n_samples
        
    def query(self, save_fn):
        projects = self.projection_collection.find()
        
        cnt = 0
        list_projects = []
        
        for project in tqdm(projects, total=self.n_samples):
            if 'category' not in project: continue
            list_projects.append(project)
            
            if cnt > self.n_samples: break
            cnt += 1
        
        self.save_to_json(list_projects, save_fn)
            
    def save_to_json(self, data, fn):
        with open(fn, 'w') as f:
            json.dump(data, f, indent=4)
            
        