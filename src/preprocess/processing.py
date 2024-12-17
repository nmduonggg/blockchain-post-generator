import json
import os
import utils.utils as utils


def local_input(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def main():
    folder = './local_database'
    bnb_path = os.path.join(folder, 'bnb.json')
    eth_path = os.path.join(folder, 'eth.json')
    project_path = os.path.join(folder, 'projects.json')
    
    utils.block_analyse(
        local_input(bnb_path), folder='./analysis/bnb'
    )
    utils.block_analyse(
        local_input(eth_path), folder='./analysis/eth'
    )
    utils.projects_analyse(
        local_input(project_path), folder='./analysis/projects'
    )
    
if __name__=='__main__':
    main()