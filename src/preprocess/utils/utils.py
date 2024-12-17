import os
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from textwrap import wrap

import utils.highlevel as highlevel

def to_datetime(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M').replace(' ', '\n')

def add_folder(folder, path):
    return os.path.join(folder, path)

def save_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)

def save_line_plot(x, y, fn, x_label, y_label, title):
    
    # Create a line plot
    plt.plot(range(len(x)), y)

    # Add labels and title
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=25)

    # Set the x-ticks and labels to the selected 10 points
    # Select 10 evenly spaced points from the x-axis
    
    if len(x) > 10:
        tick_indices = np.linspace(0, len(x) - 1, 10, dtype=int)  # 10 equally spaced indices
        tick_labels = [to_datetime(int(x[i])) for i in tick_indices]  # Get the corresponding datetime labels
        plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=12)  # Rotate for better readability
    else:
        x_dates = [to_datetime(int(i)) for i in x]
        plt.xticks(ticks=range(len(x)), labels=x_dates, rotation=45, ha='right', fontsize=12)

    # Optional: Add grid lines
    # plt.grid(True)

    # Set tick parameters for both axes
    plt.tick_params(axis='both', labelsize=20)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png', dpi=300)  # Save as PNG with high resolution
    plt.cla()  # Clear the plot to avoid interference with subsequent plots

def save_line_forecast_plot(x, y, n, fn, x_label, y_label, title, nticks=15):
    
    # Create a line plot
    
    x_old = x[:-n]
    y_old = y[:-n]
    x_pred = x[-n:]
    y_pred = y[-n:]
    
    plt.plot(range(len(x_old)), y_old, color='blue', label='Original data')
    plt.plot([i + len(x_old) for i in range(len(x_pred))], y_pred, color='red', label='Prediction')

    # Add labels and title
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=25)
    
    plt.legend()

    # Set the x-ticks and labels to the selected 10 points
    # Select 10 evenly spaced points from the x-axis
    
    if len(x) > 10:
        tick_indices = np.linspace(0, len(x) - 1, nticks, dtype=int)  # 10 equally spaced indices
        tick_labels = [to_datetime(int(x[i])) for i in tick_indices]  # Get the corresponding datetime labels
        plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=12)  # Rotate for better readability
    else:
        x_dates = [to_datetime(int(i)) for i in x]
        plt.xticks(ticks=range(len(x)), labels=x_dates, rotation=45, ha='right', fontsize=12)

    # Optional: Add grid lines
    # plt.grid(True)

    # Set tick parameters for both axes
    plt.tick_params(axis='both', labelsize=20)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png', dpi=300)  # Save as PNG with high resolution
    plt.cla()  # Clear the plot to avoid interference with subsequent plots

    
def save_scatter_plot(x, y, fn,
                   x_label, y_label, title):
    # Create a line plot
    # plt.scatter(x, y, color='blue', marker='o')
    plt.scatter(range(len(x)), y)

    # Add labels and title
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=25)

    # Set the x-ticks and labels to the selected 10 points
    if len(x) > 10:
        tick_indices = np.linspace(0, len(x) - 1, 10, dtype=int)  # 10 equally spaced indices
        tick_labels = [to_datetime(int(x[i])) for i in tick_indices]  # Get the corresponding datetime labels
        plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=12)  # Rotate for better readability
    else:
        x_dates = [to_datetime(int(i)) for i in x]
        plt.xticks(ticks=range(len(x)), labels=x_dates, rotation=45, ha='right', fontsize=12)
        
    plt.tick_params(axis='both', labelsize=15)

    # Optional: Add grid lines
    # plt.grid(True)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png')  # Save as PNG with high resolution (300 dpi)
    plt.cla()
    
def save_bar_plot_utc(x, y, fn,
                   x_label, y_label, title):
    # Binning timestamps
    # Step 1: Create 10 bins using linspace (or use histogram for dynamic binning)
    n_bins = 10
    bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)  # Create 10 bins

    # Step 2: Use digitize to categorize data into the bins
    bin_indices = np.digitize(x, bin_edges)  # Assign each timestamp to a bin
    binned_ys = []
    binned_xs = []
    for bin in range(1, n_bins+1):
        bin_mask = (bin_indices==bin).astype('bool')
        y_masked = y[bin_mask]
        x_masked = x[bin_mask]
        
        binned_xs.append(np.mean(x_masked))
        binned_ys.append(y_masked)
    
    labels = [to_datetime(int(bx)) for bx in binned_xs]

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    bplot = ax.boxplot(binned_ys,
                    patch_artist=False,  # fill with color
                    tick_labels=labels)  # will be used to label x-ticks


    # Add labels and title
    plt.xlabel(x_label, fontsize=15)  # Label for the x-axis
    plt.ylabel(y_label, fontsize=15)  # Label for the y-axis
    plt.title(title, fontsize=20)  # Title of the plot
    
    plt.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45)

    # Optional: Add grid lines
    # plt.grid(True)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png')  # Save as PNG with high resolution (300 dpi)
    plt.cla()
    
def save_bar_plot(x, y, fn,
                   x_label, y_label, title):

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    ax.bar(x, y)

    # Add labels and title
    plt.xlabel(x_label, fontsize=15)  # Label for the x-axis
    plt.ylabel(y_label, fontsize=15)  # Label for the y-axis
    plt.title(title, fontsize=20)  # Title of the plot
    
    plt.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45)

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png')  # Save as PNG with high resolution (300 dpi)
    plt.cla()
    
    
###---------- Block Analysis ----------###

def block_analyse(blocks, folder='./'):
    # blocks are filtered blocks queried from db
    os.makedirs(folder, exist_ok=True)
    
    times = []
    transaction_cnt = []
    gas_used = []
    
    for block in blocks:
        
        times.append(block['timestamp'])
        transaction_cnt.append(block['transaction_count'])
        gas_used.append(block['gas_used'])
        
    times = np.array(times)
    transaction_cnt = np.array(transaction_cnt)
    gas_used = np.array(gas_used).astype('float')
    
    save_bar_plot_utc(
        x = times,
        y = gas_used,
        fn = add_folder(folder, 'gas_used_per_block_(bar).png'),
        x_label = 'Date & Time',
        y_label='Gas Used (log10)',
        title='Gas used in block through time'
    )
    
    save_bar_plot_utc(
        x = times,
        y = transaction_cnt,
        fn = add_folder(folder, 'transaction_count_per_block_(bar).png'),
        x_label = 'Date & Time',
        y_label='# Transactions',
        title='Transaction Count through time'
    )
    
    save_scatter_plot(
        x = times,
        y = transaction_cnt,
        fn = add_folder(folder, 'transaction_count_per_block_(scatter).png'),
        x_label = 'Date & Time',
        y_label='# Transactions',
        title='Transaction Count through time'
    )
    
    gas_used = np.log10(gas_used)
    save_scatter_plot(
        x = times,
        y = gas_used,
        fn = add_folder(folder, 'gas_used_per_block_(scatter).png'),
        x_label = 'Date & Time',
        y_label='Gas Used (log10)',
        title='Gas used in block through time'
    )
    
    return

###---------- Project Analysis ----------###

def visualize_category_num(categories_dict, fn, x_label, y_label, title):
    nums = [v['num_instances'] for v in categories_dict.values()]
    labels = categories_dict.keys()
    
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    ax.bar(labels, nums)

    # Add title and labels
    plt.xlabel(x_label)  # Label for the x-axis
    plt.ylabel(y_label)  # Label for the y-axis
    plt.title(title)  # Title of the plot
    plt.xticks(rotation=45)
    
    # Save the plot as an image (e.g., PNG format)
    plt.savefig(fn, format='png')  # Save as PNG with high resolution (300 dpi)
    plt.cla()

def sort_categories_by_num(projects):
    categories = {}
    idx = 0
    for project in projects:
        idx += 1
        
        category = project['category']
        cate_item = categories.get(category, {})
        
        current_keys = cate_item.get('key', set())
        cate_data = {
            'keys': current_keys.union(set(project.keys())) if len(current_keys)==0 else \
                current_keys.intersection(set(project.keys())),
            'num_instances': cate_item.get('num_instances', 0) + 1
        }
        categories[category] = cate_data
        
    categories ={k: v for k, v in sorted(categories.items(), key=lambda x: x[1]['num_instances'], reverse=True)}
    
    return categories

def filter_NFTs_DeFi(projects):
    # NFT has 'price'
    # DeFi has 'tvl'
    nfts, defis = [], []
    for project in projects:
        if 'price' in project: nfts.append(project)
        if 'tvl' in project: defis.append(project)

    return nfts, defis

########## NFT Holder ##########
class NFT():
    def __init__(self, nfts_list, sort=True):
        super().__init__()
        self.nfts = nfts_list
        if sort: self._sort()
        
    def _sort(self, property="price"):
        filtered_nfts = [n for n in self.nfts if n[property] is not None]
        sorted_nfts = sorted(filtered_nfts, key=lambda x: int(x.get(property, 0)), reverse=True)
        return sorted_nfts
    
    def get_top_k_price(self, k):
        top_k_price = self._sort('price')[:k]
        return top_k_price
    
    def get_top_k_volume(self, k):
        top_k_volume = self._sort('volume')[:k]
        return top_k_volume
    
    def get_price_change_logs_of_best(self):
        top_k_price = self.get_top_k_price(k=1)[0]
        name = top_k_price['name']
        change_log = top_k_price['priceChangeLogs']
    
        timestamps, prices = [], []
        for timestamp, price in change_log.items():
            timestamps.append(int(timestamp))
            prices.append(float(price))
    
        return name, timestamps, prices
    
    def get_highest_price_insights(self):
        
        analyse_dict = {}
        
        top_k_price = self.get_top_k_price(k=5)
        for nft in top_k_price:
            name = nft['name']
            change_log = nft['priceChangeLogs']
            
            timestamps, prices, deltas = [], [], []
            for i, (timestamp, price) in enumerate(change_log.items()):
                prices.append(float(price))
                timestamps.append(int(timestamp))
                if i > 0: 
                    delta = prices[i] - prices[i-1]
                    deltas.append(delta)
    
            if all(x >= 0 for x in deltas): trend='increase'
            elif all(x < 0 for x in deltas): trend='decrease'
            else: trend='fluctuate'
            
            analyse_dict[name] = {
                'description': nft.get('description', ''),
                'trend': trend,
                'timestamps': timestamps,
                'prices': prices,
                'mean': np.mean(prices),
                'max': max(prices),
                'min': min(prices)
            }
            
        return analyse_dict

    def get_volume_change_logs_of_best(self):
        top_k_vol = self.get_top_k_volume(k=1)[0]
        name = top_k_vol['name']
        change_log = top_k_vol['volumeChangeLogs']
        
        timestamps, volumes = [], []
        for timestamp, volume in change_log.items():
            timestamps.append(int(timestamp))
            volumes.append(int(volume))
            
        return name, timestamps, volumes
    
    def get_highest_volume_insights(self):
        
        analyse_dict = {}
        
        top_k_price = self.get_top_k_volume(k=5)
        for nft in top_k_price:
            name = nft['name']
            change_log = nft['volumeChangeLogs']
            
            timestamps, volumes, deltas = [], [], []
            for i, (timestamp, price) in enumerate(change_log.items()):
                volumes.append(float(price))
                timestamps.append(int(timestamp))
                if i > 0: 
                    delta = volumes[i] - volumes[i-1]
                    deltas.append(delta)
    
            if all(x >= 0 for x in deltas): trend='increase'
            elif all(x < 0 for x in deltas): trend='decrease'
            elif len(deltas)==0: trend='constant'
            else: trend='fluctuate'
            
            analyse_dict[name] = {
                'description': nft.get('description', ''),
                'trend': trend,
                'timestamps': timestamps,
                'volumes': volumes,
                'mean': np.mean(volumes),
                'max': max(volumes),
                'min': min(volumes)
            }
            
        return analyse_dict
        
##########################

########## DeFi Holder ##########
class DeFi():
    def __init__(self, defis_list, sort=True):
        super().__init__()
        self.defis = defis_list
        if sort: self._sort()
        
    def _sort(self, property="tvl"):
        filtered_defis = [n for n in self.defis if n[property] is not None]
        sorted_defis = sorted(filtered_defis, key=lambda x: int(x.get(property, 0)), reverse=True)
        return sorted_defis
    
    def get_top_k_tvl(self, k):
        top_k_price = self._sort('tvl')[:k]
        return top_k_price
    
    def get_tvl_change_logs_of_best(self):
        top_k_price = self.get_top_k_tvl(k=1)[0]
        name = top_k_price['name']
        change_log = top_k_price['tvlChangeLogs']
    
        timestamps, tvls = [], []
        for timestamp, tvl in change_log.items():
            timestamps.append(int(timestamp))
            tvls.append(float(tvl))
    
        return name, timestamps, tvls
    
    def get_highest_tvl_insights(self):
        
        analyse_dict = {}
        
        top_k_price = self.get_top_k_tvl(k=5)
        for nft in top_k_price:
            name = nft['name']
            change_log = nft['tvlChangeLogs']
            
            timestamps, tvls, deltas = [], [], []
            for i, (timestamp, tvl) in enumerate(change_log.items()):
                tvls.append(float(tvl))
                timestamps.append(int(timestamp))
                if i > 0: 
                    delta = tvls[i] - tvls[i-1]
                    deltas.append(delta)
    
            if all(x >= 0 for x in deltas): trend='increase'
            elif all(x < 0 for x in deltas): trend='decrease'
            elif len(deltas)==0: trend='constant'
            else: trend='fluctuate'
            
            
            analyse_dict[name] = {
                'description': nft.get('description', ''),
                'trend': trend,
                'timestamps': timestamps,
                'tvls': tvls,
                'mean': np.mean(tvls),
                'max': max(tvls),
                'min': min(tvls)
            }
            
        return analyse_dict
    
##########################

def projects_analyse(projects, k=10, folder='./'):
    os.makedirs(folder, exist_ok=True)
    all_key_profiles = list()
    
    sorted_categories = sort_categories_by_num(projects)
    
    cnt = 0
    top_k_categories = {}
    for cls, data in sorted_categories.items():
        if cnt==k-1: break
        top_k_categories[cls] = data
        cnt += 1
    
    visualize_category_num(top_k_categories,
                           fn = add_folder(folder, 'number_of_project_in_each_category.png'),
                           x_label = 'Project Category',
                           y_label = 'Number of projects',
                           title = 'Capacity of different categories')
    
    nfts, defis = filter_NFTs_DeFi(projects)
    
    
    ##--- NFT processing ---##
    nft_holder = NFT(nfts)
    
    # top k price nft
    top_k_price = nft_holder.get_top_k_price(k=k)
    names = [data['name'] for data in top_k_price]
    prices = [float(data['price']) for data in top_k_price]
    save_bar_plot(x=names, y=prices, fn=add_folder(folder, 'price_of_top_NFTs.png'),
                   x_label='NFT name', y_label='Price', title='Price of top NFTs')
    
    # top k volume nft
    top_k_volume = nft_holder.get_top_k_volume(k=k)
    names = [data['name'] for data in top_k_volume]
    volumes = [float(data['volume']) for data in top_k_volume]
    save_bar_plot(x=names, y=volumes, fn=add_folder(folder, 'volume_of_top_NFTs.png'),
                   x_label='NFT name', y_label='Volume', title='Volume of top NFTs')
    
    # price change log of the most expensive NFT
    price_changelog_best = nft_holder.get_price_change_logs_of_best()
    name, timestamps, prices = price_changelog_best
    
    #-- arima prediction
    n_predictions, new_timestamps, new_prices = highlevel.arima_pipeline(timestamps, prices)
    save_line_forecast_plot(x=new_timestamps, y=new_prices, n=n_predictions,
                            fn=add_folder(folder, f"ARIMA_price_prediction_of_best_NFT_named_{name}.png"),
                            x_label="Date & Time", y_label="Price",
                            title=f"ARIMA price prediction of the most expensive NFT named {name}")
    
    #-- change log
    save_line_plot(x=timestamps, y=prices, fn=add_folder(folder, f"price_change_log_of_best_NFT_named_{name}.png"),
                   x_label="Date & Time", y_label="Price",
                   title=f"Price changes across time of the most expensive NFT named {name}")
    
    
    # volume change log of NFT with the largest capacity
    volume_changelog_best = nft_holder.get_volume_change_logs_of_best()
    name, timestamps, volumes = volume_changelog_best
    #-- arima prediction
    n_predictions, new_timestamps, new_volumes = highlevel.arima_pipeline(timestamps, volumes)
    save_line_forecast_plot(x=new_timestamps, y=new_volumes, n=n_predictions,
                            fn=add_folder(folder, f'ARIMA_volume_prediction_of_best_NFT_named_{name}.png'),
                            x_label="Date & Time", y_label="Volume",
                            title=f"ARIMA Volume prediction across time of the largest NFT in volume named {name}.")
    
    # top 5 price analysis
    price_changelog_top5 = nft_holder.get_highest_price_insights()
    for name in price_changelog_top5:
        profile = price_changelog_top5[name]
        profile_path = add_folder(folder, os.path.join('profiles', 'nft', name, 'price'))
        profile['name'] = name
        profile['path'] = profile_path
        os.makedirs(profile_path, exist_ok=True)
        
        timestamps, prices = profile['timestamps'], profile['prices']
        n_predictions, new_timestamps, new_prices = highlevel.arima_pipeline(timestamps, prices)
        
        profile['predict_timestamps'] = new_timestamps[-n_predictions:]
        profile['predicted_prices'] = new_prices[-n_predictions:]
        profile['value_type'] = 'prices'
        
        save_line_forecast_plot(
            x=new_timestamps, y=new_prices, n=n_predictions,
            fn=add_folder(profile_path, f"ARIMA_price_prediction_of_NFT_named_{name}.png"),
            x_label="Date & Time", y_label="Price",
            title=f"ARIMA price prediction across time of the NFT named {name}")
        save_json(profile, fn=add_folder(profile_path, 'statisics.json'))
        
        all_key_profiles.append(profile)
    
    # top 5 volume analysis
    volume_changelog_top5 = nft_holder.get_highest_volume_insights()
    for name in volume_changelog_top5:
        profile = volume_changelog_top5[name]
        profile_path = add_folder(folder, os.path.join('profiles', 'nft', name, 'volume'))
        profile['name'] = name
        profile['path'] = profile_path
        os.makedirs(profile_path, exist_ok=True)
        
        timestamps, volumes = profile['timestamps'], profile['volumes']
        n_predictions, new_timestamps, new_volumes = highlevel.arima_pipeline(timestamps, volumes)
        profile['predict_timestamps'] = new_timestamps[-n_predictions:]
        profile['predicted_volumes'] = new_volumes[-n_predictions:]
        profile['value_type'] = 'volumes'
        save_line_forecast_plot(
            x=new_timestamps, y=new_volumes, n=n_predictions,
            fn=add_folder(profile_path, f"ARIMA_volume_prediction_of_NFT_named_{name}.png"),
            x_label="Date & Time", y_label="Volume",
            title=f"ARIMA Volume prediction across time of the NFT named {name}")
        save_json(profile, fn=add_folder(profile_path, 'statisics.json'))
        
        all_key_profiles.append(profile)
    
    ## DeFi processing ##
    defi_holder = DeFi(defis)
    top_k_tvl = defi_holder.get_top_k_tvl(k=k)
    names = [data['name'] for data in top_k_price]
    prices = [float(data['tvl']) for data in top_k_tvl]
    save_bar_plot(x=names, y=prices, fn=add_folder(folder, 'total_value_locked_of_top_DeFi_projects.png'),
                   x_label='Decentralized Finance (DeFi) projects', y_label='Total value locked (TVL)', title='Total value locked of top DeFi projects.')
    
    
    tvl_changelog_best = defi_holder.get_tvl_change_logs_of_best()
    name, timestamps, tvls = tvl_changelog_best
    
    #-- arima forecast
    n_predictions, new_timestamps, new_tvls = highlevel.arima_pipeline(timestamps, tvls)
    save_line_forecast_plot(x=new_timestamps, y=new_tvls, n=n_predictions,
                            fn=add_folder(folder, f'ARIME_total_value_locked_prediction_of_best_DeFi_project_named_{name}.png'),
                            x_label='Date & Time', y_label='Total value locked', 
                            title=f'ARIMA total value locked prediction across time of the largest DeFi project named {name}.')
    
    save_line_plot(x=timestamps, y=tvls, fn=add_folder(folder, f'total_value_locked_change_log_of_best_DeFi_project_named_{name}.png'),
                   x_label='Date & Time', y_label='Total value locked', 
                   title=f'Total value locked changes across time of the largest DeFi project named {name}.')
    
    # top 5 TVL analysis
    tvl_changelog_top5 = defi_holder.get_highest_tvl_insights()
    for name in tvl_changelog_top5:
        profile = tvl_changelog_top5[name]
        profile_path = add_folder(folder, os.path.join('profiles', 'defi', name, 'tvl'))
        profile['name'] = name
        profile['path'] = profile_path
        os.makedirs(profile_path, exist_ok=True)
        
        timestamps, tvls = profile['timestamps'], profile['tvls']
        n_predictions, new_timestamps, new_tvls = highlevel.arima_pipeline(timestamps, tvls)
        
        profile['predict_timestamps'] = new_timestamps[-n_predictions:]
        profile['predicted_tvls'] = new_tvls[-n_predictions:]
        profile['value_type'] = 'tvls'
        
        save_line_forecast_plot(
            x=new_timestamps, y=new_tvls, n=n_predictions,
            fn=add_folder(profile_path, f"ARIMA_total_value_locked_prediction_of_NFT_named_{name}.png"),
            x_label="Date & Time", y_label="Total Value Locked",
            title=f"ARIMA Total value locked prediction across time of the DeFi named {name}")
        save_line_plot(
            x=timestamps, y=tvls, fn=add_folder(profile_path, f"total_value_locked_change_log_of_NFT_named_{name}.png"),
            x_label="Date & Time", y_label="Total Value Locked",
            title=f"Total value locked changes across time of the DeFi named {name}")
        save_json(profile, fn=add_folder(profile_path, 'statisics.json'))
    
        all_key_profiles.append(profile)
        
    profile_list_folder = add_folder(folder, os.path.join('keywords'))
    os.makedirs(profile_list_folder, exist_ok=True)
    save_json(all_key_profiles, fn = add_folder(profile_list_folder, 'profile_list.json'))