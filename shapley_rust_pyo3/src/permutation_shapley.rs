///////////////////////////////////////////////////////////////
// Imports
///////////////////////////////////////////////////////////////
use crate::graph_utils::{Graph, sanity_check, read_direct_weighted_graph};
use crate::influence_model_utils::monte_carlo_simulation;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use rand::prelude::*;
use rayon::prelude::*;
use std::f64;
use std::error::Error;
use std::time::Instant;


/// Helper to turn a `HashSet<usize>` into a **sorted** `Vec<usize>`
/// so we can use it as a key in `HashMap`.
fn to_sorted_vec(s: &HashSet<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = s.iter().copied().collect();
    v.sort_unstable();
    v
}

///////////////////////////////////////////////////////////////
/// get_influence
///
/// Translated from:
///
/// ```python
/// def get_influence(S_subset, graph, max_hop, m, value_cache):
///     ...
/// ```
/// 
/// Returns the Monte Carlo–estimated influence for a given subset `S_subset`.
/// Uses a shared cache (`value_cache`) keyed by the subset's contents.
///
///////////////////////////////////////////////////////////////
pub fn get_influence(
    s_subset: &HashSet<usize>,
    graph: &Graph, 
    max_hop: Option<usize>,
    m: usize,
    value_cache: &Arc<Mutex<HashMap<Vec<usize>, f64>>>,
) -> f64 {
    // Convert the set to a sorted vector to use as a key
    let subset_key = to_sorted_vec(s_subset);

    // Try to look up in the cache
    {
        let cache_guard = value_cache.lock().unwrap();
        if let Some(&cached_val) = cache_guard.get(&subset_key) {
            return cached_val;
        }
    }

    // Otherwise, run the Monte Carlo simulation
    // (Assuming you have a function `monte_carlo_simulation(...)`
    //  in your `influence_model_utils` that matches the Python version.)
    let (mean_val, _) = /*influence_model_utils::*/monte_carlo_simulation(
        graph,
        s_subset,
        // None,        // optional RNG
        max_hop,
        m
    );

    // Insert the result into the cache
    {
        let mut cache_guard = value_cache.lock().unwrap();
        cache_guard.insert(subset_key, mean_val);
    }

    mean_val
}

///////////////////////////////////////////////////////////////
/// approximate_shapley_permutation
///
/// Translated from:
///
/// ```python
/// def approximate_shapley_permutation(
///     graph, gen_seed=None, max_hop=None,
///     n=None, m=None, epsilon=None, delta=None
/// ): ...
/// ```
/// 
/// Approximate Shapley values for each seed using
/// the permutation-based approach (Algorithm 5, single-threaded).
///
///////////////////////////////////////////////////////////////
pub fn approximate_shapley_permutation(
    graph: &Graph,
    gen_seed: Option<u64>,
    max_hop: Option<usize>,
    mut n: Option<usize>,
    mut m: Option<usize>,
    epsilon: Option<f64>,
    delta: Option<f64>,
) -> HashMap<usize, f64> {
    // 1) Basic checks (like sanity_check in Python).
    // e.g.: graph_utils::sanity_check(&graph.G, &graph.seeds, &graph.P);
    // For illustration, we call them directly if needed:
    // sanity_check(&graph.g, graph.seeds.as_ref().unwrap(), &graph.p);

    let seeds: Vec<usize> = graph.seeds.as_ref().unwrap().iter().copied().collect();
    let n_seeds = seeds.len();

    // If user did not provide n or m, compute them from epsilon, delta
    if n.is_none() || m.is_none() {
        let eps = epsilon.unwrap_or(0.5);
        let dlt = delta.unwrap_or(0.1);

        let n_calc = ((8.0 * (graph.n as f64).powi(2) / eps.powi(2))
            * (4.0 * n_seeds as f64 / dlt).ln())
            .ceil() as usize;

        let m_calc = ((8.0 * (graph.n as f64).powi(2) / eps.powi(2))
            * (4.0 * n_calc as f64 * n_seeds as f64 / dlt).ln())
            .ceil() as usize;

        n = Some(n_calc);
        m = Some(m_calc);
    }
    let n = n.unwrap();
    let m = m.unwrap();

    println!("parameters: n={}, m={}", n, m);

    // Optional seeding for reproducibility
    if let Some(seed) = gen_seed {
        rand::rngs::StdRng::seed_from_u64(seed); // note: does not globally seed thread_rng()
        // If you want to explicitly create a local RNG:
        // let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // but that is separate from global thread_rng().
    }

    // Prepare accumulators
    let mut est: HashMap<usize, f64> = seeds.iter().map(|&t| (t, 0.0)).collect();

    // Use an Arc<Mutex<...>> for the caching structure
    // (equivalent to Python's dictionary in local scope).
    let value_cache = Arc::new(Mutex::new(HashMap::<Vec<usize>, f64>::new()));

    // Single-threaded approach: sample n permutations
    let mut rng = thread_rng();
    for _ in 0..n {
        // Random permutation of seeds
        let mut perm = seeds.clone();
        perm.shuffle(&mut rng);

        let mut s_set = HashSet::new();
        for &t_j in &perm {
            let mut s_with = s_set.clone();
            s_with.insert(t_j);

            let val_s_with = get_influence(&s_with, graph, max_hop, m, &value_cache);
            let val_s = get_influence(&s_set, graph, max_hop, m, &value_cache);
            let delta = val_s_with - val_s;
            *est.get_mut(&t_j).unwrap() += delta;

            s_set = s_with;
        }
    }

    // Average after n permutations
    let mut shap_approx = HashMap::new();
    for &t in &seeds {
        shap_approx.insert(t, est[&t] / n as f64);
    }
    shap_approx
}

///////////////////////////////////////////////////////////////
/// process_permutation
///
/// Translated from:
///
/// ```python
/// def process_permutation(args):
///     ...
/// ```
/// 
/// In Python, it takes a single tuple. Here, we can make it a normal function
/// that returns local marginal contributions for one random permutation.
///
///////////////////////////////////////////////////////////////
pub fn process_permutation(
    graph: &Graph,
    seeds: &[usize],
    max_hop: Option<usize>,
    m: usize,
    shared_cache: &Arc<Mutex<HashMap<Vec<usize>, f64>>>,
) -> HashMap<usize, f64> {
    let mut rng = thread_rng();
    let mut perm = seeds.to_vec();
    perm.shuffle(&mut rng);

    let mut local_est: HashMap<usize, f64> = seeds.iter().map(|&t| (t, 0.0)).collect();

    let mut s_set = HashSet::new();
    for &t_j in &perm {
        let mut s_with = s_set.clone();
        s_with.insert(t_j);

        let val_s_with = get_influence(&s_with, graph, max_hop, m, shared_cache);
        let val_s = get_influence(&s_set, graph, max_hop, m, shared_cache);
        let delta_val = val_s_with - val_s;
        *local_est.get_mut(&t_j).unwrap() += delta_val;

        s_set = s_with;
    }
    local_est
}

///////////////////////////////////////////////////////////////
/// approximate_shapley_permutation_parallel
///
/// Translated from:
///
/// ```python
/// def approximate_shapley_permutation_parallel(...):
///     ...
/// ```
/// 
/// Uses parallel iteration (e.g., via rayon) to process multiple random
/// permutations concurrently, and returns approximate Shapley values.
///
///////////////////////////////////////////////////////////////
pub fn approximate_shapley_permutation_parallel(
    graph: &Graph,
    parallel: bool,
    gen_seed: Option<u64>,
    max_hop: Option<usize>,
    mut n: Option<usize>,
    mut m: Option<usize>,
    epsilon: Option<f64>,
    delta: Option<f64>,
    n_jobs: Option<usize>,
) -> HashMap<usize, f64> {
    // Configure Rayon thread pool to use at most 48 cores
    // This is set once for the entire process
    rayon::ThreadPoolBuilder::new()
        .num_threads(48)
        .build_global()
        .unwrap_or_else(|e| {
            eprintln!("Warning: Failed to configure thread pool: {}", e);
        });
        
    // Similar sanity checks
    // e.g.: sanity_check(&graph.g, &graph.seeds.as_ref().unwrap(), &graph.p);

    let seeds: Vec<usize> = graph.seeds.as_ref().unwrap().iter().copied().collect();
    let n_seeds = seeds.len();

    // If n or m is not set, compute from epsilon, delta
    if n.is_none() || m.is_none() {
        let eps = epsilon.unwrap_or(0.5);
        let dlt = delta.unwrap_or(0.1);

        let n_calc = ((8.0 * (graph.n as f64).powi(2) / eps.powi(2))
            * (4.0 * n_seeds as f64 / dlt).ln())
            .ceil() as usize;

        let m_calc = ((8.0 * (graph.n as f64).powi(2) / eps.powi(2))
            * (4.0 * n_calc as f64 * n_seeds as f64 / dlt).ln())
            .ceil() as usize;

        n = Some(n_calc);
        m = Some(m_calc);
    }
    let n = n.unwrap();
    let m = m.unwrap();

    println!("parameters: n = {}, m = {}", n, m);

    // Optional seeding for reproducibility
    if let Some(seed) = gen_seed {
        // This does not seed the global thread_rng();
        // it only creates a seeded RNG if you want to use it.
        let _ = rand::rngs::StdRng::seed_from_u64(seed);
    }

    // Decide how many worker threads: cannot exceed n or 48
    let n_jobs = n_jobs.unwrap_or_else(|| 
        // Use the minimum of user-specified threads, available threads, and 48
        rayon::current_num_threads().min(48)
    ).min(n);

    // Build a shared cache using Arc<Mutex<...>> so each thread can read/write safely
    let shared_cache = Arc::new(Mutex::new(HashMap::<Vec<usize>, f64>::new()));

    // Build tasks: we have n permutations total
    let tasks: Vec<_> = (0..n)
        .map(|_| (graph, seeds.clone(), max_hop, m, Arc::clone(&shared_cache)))
        .collect();

    // Each task produces a local_est
    // We accumulate them in parallel using rayon
    // let results: Vec<HashMap<usize, f64>> = tasks
    //     .par_iter()
    //     .map(|(g, s, mh, mm, sc)| {
    //         process_permutation(g, s, *mh, *mm, sc)
    //     })
    //     .collect();


    // Define a closure that processes one task
    let process_task = |(g, s, mh, mm, sc): &(
        &Graph,
        Vec<usize>,
        Option<usize>,
        usize,
        Arc<Mutex<HashMap<Vec<usize>, f64>>>,
    )| {
        process_permutation(g, s, *mh, *mm, sc)
    };

    // Run tasks either in parallel or single-threaded
    let results: Vec<HashMap<usize, f64>> = if parallel {
        tasks.par_iter().map(process_task).collect()
    } else {
        tasks.iter().map(process_task).collect()
    };

    
    // Combine results
    let mut total_est: HashMap<usize, f64> = seeds.iter().map(|&t| (t, 0.0)).collect();
    for local_est in results {
        for &t in &seeds {
            *total_est.get_mut(&t).unwrap() += local_est[&t];
        }
    }

    // Average over n permutations
    let mut shap_approx = HashMap::new();
    for &t in &seeds {
        shap_approx.insert(t, total_est[&t] / n as f64);
    }
    shap_approx
}
