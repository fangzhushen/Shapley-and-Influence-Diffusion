import shapley_rust
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm
import csv
from driver_rust import run_algorithm





# Create directories if they do not exist
os.makedirs('../results/twitter_parallel/shapley_values', exist_ok=True)
os.makedirs('../results/twitter_parallel/raw_logs', exist_ok=True)

def process_one_run(args):
    folder_path = 'twitter'
    seed_select_method, num_seeds, algo, parameters = args
    print(f"Starting run: seed_select_method={seed_select_method}, num_seeds={num_seeds}, algo={algo}")
    
    # Load the graph
    graph_path = f'../data/{folder_path}/{folder_path}_{num_seeds}{seed_select_method}seeds.bin'
    #print(f"Loading graph from {graph_path}")
    graph = shapley_rust.PyGraph.load_from_file(graph_path)
    if parameters['parallel']:
        folder_path = 'twitter_parallel'
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

if __name__ == '__main__':
    
    seed_select_method = 'deg'
    
    num_seeds_ls = [10] 

    algo_ls_multi_step = ['rr','permute1','permute2']
    algo_ls_single_step = ['permute1single','permute2single','rrsingle','dp_single_optimize']

    algo_ls = ['rr']

    # Example random seeds for the experiments
    gen_seed_ls = [1]

    # Additional parameters for each algorithm remain the same as in the dictionary
    run_all = True
    parallel = True

    # Build the list of runs
    if run_all:
        all_combinations = []   
        for algo in algo_ls:
            if algo == 'permute1':
                # For demonstration: vary 'm' and 'n' as you want; here is an example
                n = 500
                m = 500
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {'n': n, 'm': m, 'gen_seed': gen_seed, 'parallel': parallel}
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                )
            elif algo == 'permute1single':
                # For demonstration: vary 'm' and 'n' as you want; here is an example
                n = 500
                m = 500
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {'n': n, 'm': m, 'gen_seed': gen_seed, 'parallel': parallel, 'max_hop': 1}
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                )
            elif algo == 'permute2':
                n = 5000
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {'n': n, 'gen_seed': gen_seed, 'parallel': parallel}
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                    )
            elif algo == 'permute2single':
                n = 5000
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {'n': n, 'gen_seed': gen_seed, 'parallel': parallel, 'max_hop': 1}
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                    )
            elif algo == 'rr':
                # Example parameters
                k = 5
                epsilon = 0.1
                ell = 1
                num_theta = 500000
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {
                        'k': k,
                        'epsilon': epsilon,
                        'ell': ell,
                        'num_theta': num_theta,
                        'gen_seed': gen_seed,
                        'parallel': parallel
                    }
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                    )
            elif algo == 'rrsingle':
                # Example parameters
                k = 5
                epsilon = 0.1
                ell = 1
                num_theta = 500000
                for num_seeds, gen_seed in product(num_seeds_ls, gen_seed_ls):
                    parameters = {
                        'k': k,
                        'epsilon': epsilon,
                        'ell': ell,
                        'num_theta': num_theta,
                        'gen_seed': gen_seed,
                        'parallel': parallel,
                        'max_hop': 1
                    }
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                    )
            elif algo == 'dp_single_optimize':
                for num_seeds in num_seeds_ls:
                    parameters = {'gen_seed': 1,'parallel': parallel}
                    all_combinations.append(
                        (seed_select_method, num_seeds, algo, parameters)
                    )
        if not parallel:
            with Pool(processes=os.cpu_count()) as pool:
                results = list(tqdm(
                    pool.imap(process_one_run, all_combinations),
                    total=len(all_combinations),
                    desc="Processing facebook experiments"
                ))
        else:
            for combination in all_combinations:
                process_one_run(combination)

    else:
        #single run for ground truth  
        algo = 'rr'
        if algo == 'permute1':
            parameters = {'n': 500, 'm': 500, 'gen_seed': 42, 'parallel': True}
        elif algo == 'permute2':
            parameters = {'n': 5000, 'gen_seed': 42, 'parallel': True}
        elif algo == 'rr':
            for num_seeds in num_seeds_ls:
                parameters = {'k': num_seeds, 'epsilon': 0.01, 'ell': 1,'gen_seed': 42, 'parallel': True}
                process_one_run((seed_select_method, num_seeds, algo, parameters))
       
       