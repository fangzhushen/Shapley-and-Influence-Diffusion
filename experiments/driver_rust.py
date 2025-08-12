
import shapley_rust 
# The above import assumes your Rust code is compiled into a Python module
# named `shapley_influence_rust`. Adjust to match your setup.

# Example driver function that selects which Rust function to call
# based on the specified algorithm and parameters.

def run_algorithm(graph, algo, parameters):
    """
    Run the specified algorithm from the Rust implementation
    """
    if algo == 'permute1' or algo == "permute1single":
        return shapley_rust.approx_permutation1(
            graph=graph,
            parallel=parameters.get('parallel', True),
            gen_seed=parameters.get('gen_seed', None),
            max_hop=parameters.get('max_hop', None),
            n=parameters.get('n', None),
            m=parameters.get('m', None),
            epsilon=parameters.get('epsilon', None),
            delta=parameters.get('delta', None),
            n_jobs=parameters.get('n_jobs', None)
        )
    elif algo == 'permute2' or algo == "permute2single":
        return shapley_rust.approx_permutation2(
            graph=graph,
            gen_seed=parameters.get('gen_seed', None),
            max_hop=parameters.get('max_hop', None),
            n=parameters.get('n', None),
            epsilon=parameters.get('epsilon', None),
            delta=parameters.get('delta', None),
            parallel=parameters.get('parallel', True)
        )
    elif algo == 'rr' or algo == "rrsingle":
        return shapley_rust.rr_shapley_approx(
            graph=graph,
            k=parameters.get('k', 1),
            eps=parameters.get('epsilon', 0.1),
            ell=parameters.get('ell', 1),
            random_seed=parameters.get('gen_seed', None),
            num_theta=parameters.get('num_theta', None),
            parallel=parameters.get('parallel', True),
            max_hop=parameters.get('max_hop', None)
        )
    elif algo == 'dp_single':
        return shapley_rust.single_step_shapley_rust(graph)   
    elif algo == 'dp_single_optimize':
        return shapley_rust.single_step_shapley_optimize_rust(graph)
    elif algo == 'exact_single':
        return shapley_rust.exact_single_step_shapley_rust(graph)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
