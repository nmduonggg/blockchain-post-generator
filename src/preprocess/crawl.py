import os
import crawler
from datetime import datetime, timedelta

def main():
    
    now = datetime.now()
    prev_days = 1
    
    
    save_folder = './local_database'
    os.makedirs(save_folder, exist_ok=True)
    
    bnb_crawler = crawler.BlocksCrawler('blockchain_etl', 50000)
    bnb_crawler.query_from_time(now, prev_days, save_fn=os.path.join(save_folder, 'bnb.json'))
    
    eth_crawler = crawler.BlocksCrawler('ethereum_blockchain_etl', 50000)
    eth_crawler.query_from_time(now, prev_days, save_fn=os.path.join(save_folder, 'eth.json'))
    
    project_crawler = crawler.ProjectsCrawler(50000)
    project_crawler.query(save_fn=os.path.join(save_folder, 'projects.json'))
    
if __name__=='__main__':
    main()