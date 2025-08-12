import sys
import os
import time
import csv
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm
# Add project root to path
sys.path.append('.')
from driver_rust import run_algorithm

import shapley_rust
#add the path src
sys.path.append('./src')


# Create directories if they do not exist
os.makedirs('../results/congress_parallel/shapley_values', exist_ok=True)
os.makedirs('../results/congress_parallel/raw_logs', exist_ok=True)

# List of k values for seed selection

def process_one_run(args):
    folder_path = 'congress'
    seed_select_method, num_seeds, algo, parameters = args
    print(f"Starting run: seed_select_method={seed_select_method}, num_seeds={num_seeds}, algo={algo}")
    
    # Load the graph
    graph_path = f'../data/{folder_path}/{folder_path}_{num_seeds}{seed_select_method}seeds.bin'
    #print(f"Loading graph from {graph_path}")
    graph = shapley_rust.PyGraph.load_from_file(graph_path)
    if parameters['parallel']:
        folder_path = 'congress_parallel'
    # Construct file paths for results
    if algo in ['exact_shapley_baseline','exact_single','dp_single','dp_single_optimize']:
        shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}.csv'
        log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}.txt'
    else:
        if algo == 'permute1' or algo == 'permute1single':
            shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_m{parameters["m"]}_round{parameters["gen_seed"]}.csv'
            log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_m{parameters["m"]}_round{parameters["gen_seed"]}.txt'
        elif algo == 'permute2' or algo == 'permute2single':
            shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_round{parameters["gen_seed"]}.csv'
            log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_round{parameters["gen_seed"]}.txt'
        elif algo == 'rr' or algo == 'rrsingle':
            if 'num_theta' in parameters:
                shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["num_theta"]}_round{parameters["gen_seed"]}.csv'
                log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["num_theta"]}_round{parameters["gen_seed"]}.txt'
            else:
                shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_k{parameters["k"]}_epsilon{parameters["epsilon"]}_ell{parameters["ell"]}_round{parameters["gen_seed"]}.csv'
                log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_k{parameters["k"]}_epsilon{parameters["epsilon"]}_ell{parameters["ell"]}_round{parameters["gen_seed"]}.txt'
    
    start_time = time.time()
    results = run_algorithm(graph, algo, parameters)
    end_time = time.time()
    runtime = end_time - start_time
    results = {k: round(v, 4) for k, v in results.items()}

    # Write Shapley values to CSV
    with open(shapley_filename, 'w', newline='') as csvfile:
        fieldnames = ['seed node', 'shapley value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node, value in results.items():
            writer.writerow({
                'seed node': node,
                'shapley value': value
            })

    # Write runtime info to txt log
    with open(log_filename, 'w') as f:
        f.write(f"algorithm: {algo}\n")
        f.write(f"runtime: {runtime:.4f}\n")
        f.write(f"seed_selection_method: {seed_select_method}\n")
        f.write(f"num_seeds: {num_seeds}\n")
        for param_name, param_value in parameters.items():
            f.write(f"{param_name}: {param_value}\n")

    print(f'finished running {algo}: {runtime:.4f} seconds')
    return results

# def process_one_run(args):
#     folder_path = 'congress'
#     seed_select_method, num_seeds, algo, parameters = args
#     print(f"Starting run: seed_select_method={seed_select_method}, num_seeds={num_seeds}, algo={algo}")
    
#     # Load the graph
#     graph_path = f'../data/{folder_path}/{folder_path}_{num_seeds}{seed_select_method}seeds.bin'
#     graph = shapley_rust.PyGraph.load_from_file(graph_path)
    

#     folder_path = 'congress_test'
#     #create folder if not exists
#     os.makedirs(f'../results/{folder_path}/shapley_values', exist_ok=True)
#     os.makedirs(f'../results/{folder_path}/raw_logs', exist_ok=True)
#     if algo in ['exact_shapley_baseline','exact_single','dp_single']:
#         shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}.csv'
#         log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}.txt'
#     else:
#         if algo == 'permute1':
#             shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_m{parameters["m"]}_round{parameters["gen_seed"]}.csv'
#             log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_m{parameters["m"]}_round{parameters["gen_seed"]}.txt'
#         elif algo == 'permute2':

#                 shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_round{parameters["gen_seed"]}.csv'
            
#                 log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["n"]}_round{parameters["gen_seed"]}.txt'
           
#         elif algo == 'rr':
#             if 'num_theta' in parameters:
#                 print("1")
#                 shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["num_theta"]}_round{parameters["gen_seed"]}.csv'
#                 log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_n{parameters["num_theta"]}_round{parameters["gen_seed"]}.txt'
#             else:
#                 print("2")
#                 shapley_filename = f'../results/{folder_path}/shapley_values/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_k{parameters["k"]}_epsilon{parameters["epsilon"]}_ell{parameters["ell"]}_round{parameters["gen_seed"]}.csv'
#                 log_filename = f'../results/{folder_path}/raw_logs/{folder_path}_graph_{num_seeds}seeds_{seed_select_method}_{algo}_k{parameters["k"]}_epsilon{parameters["epsilon"]}_ell{parameters["ell"]}_round{parameters["gen_seed"]}.txt'
#     start_time = time.time()
#     results = run_algorithm(graph, algo,parameters)
#     end_time = time.time()
#     runtime = end_time - start_time
#     results = {k: round(v, 4) for k, v in results.items()}

#     # Write Shapley values to CSV
#     with open(shapley_filename, 'w', newline='') as csvfile:
#         fieldnames = ['seed node', 'shapley value']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for node, value in results.items():
#             writer.writerow({
#                 'seed node': node,
#                 'shapley value': value
#             })

#     # Write runtime info to txt log
#     with open(log_filename, 'w') as f:
#         f.write(f"algorithm: {algo}\n")
#         f.write(f"runtime: {runtime:.4f}\n")
#         f.write(f"seed_selection_method: {seed_select_method}\n")
#         f.write(f"num_seeds: {num_seeds}\n")
#         for param_name, param_value in parameters.items():
#             f.write(f"{param_name}: {param_value}\n")
#         # f.write(f"k: {k}\n")
#         # f.write(f"gen_seed: {gen_seed}\n")
#         #f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#     print(f'finished running {algo}: {runtime:.4f} seconds')
#     return results
    

def run_multi_step():
    folder_path = 'congress'
    seed_select_method = 'deg' #{deg,im}
    num_seeds_ls = [5]#[5, 10, 50]
    algo = 'rr'


    parameters = {'gen_seed': 1, 'parallel': True,'k':5, 'epsilon':0.01, 'ell':1}
    for num_seeds in num_seeds_ls:
        process_one_run((seed_select_method, num_seeds, algo, parameters))
    


    
if __name__ == '__main__':
    run_multi_step()

