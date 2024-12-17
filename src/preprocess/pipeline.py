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



db = client["blockchain_etl"]
block_collection = db['blocks']
transaction_collection = db['transactions']

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
    
    n_samples = 50000
    cnt = 0
    # block querying
    print("Start finding blocks")
    blocks = block_collection.find(query)
    list_blocks = []
    list_trans = []
    for block in tqdm(blocks, total=n_samples):
        # block_hash = block['hash']
        # blk_query = query_trans_in_block_hash(block_hash, now, month_ago)
        # transactions_in_blk = transaction_collection.find(blk_query)
        
        # print("ckpt 1")
        list_blocks.append(block)
        # list_trans += list(transactions_in_blk)
        
        if cnt==n_samples: break
        cnt += 1
        
        
    print("Saving...")
    with open('./list_blocks.json', 'w') as f:
        json.dump(list_blocks, f, indent=4)
    # with open('./list_trans.json', 'w') as f:
    #     json.dump(list_trans, f, indent=4)
    
    return blocks, list_trans

def demo_input():
    with open('./list_blocks.json', 'r') as f:
        data = json.load(f)
    return data

documents = demo_input()

utils.transaction_cnt_time(documents)

# blocks, transactions = input()