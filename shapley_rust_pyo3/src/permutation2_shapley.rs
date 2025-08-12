use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::f64;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

use crate::graph_utils::{Graph, sanity_check};

//////////////////////////////////////////////////////////////
/// sample_live_edge_graph
///
/// We iterate over each edge in `graph.p`, flip a coin with
/// probability = edge weight, and if success, add (u->v) to the
/// returned adjacency map. This is analogous to the
/// `__sample_live_edge_graph` in Python.
//////////////////////////////////////////////////////////////
pub fn sample_live_edge_graph(graph: &Graph, gen_seed: Option<u64>) -> HashMap<usize, Vec<usize>> {
    let mut rng: StdRng = match gen_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut sampled_graph: HashMap<usize, Vec<usize>> = HashMap::new();
    if let Some(prob_map) = &graph.p {
        for (&(u, v), &prob) in prob_map {
            let coin: f64 = rng.gen();
            if coin < prob {
                sampled_graph.entry(u).or_insert_with(Vec::new).push(v);
            }
        }
    }
    sampled_graph
}

//////////////////////////////////////////////////////////////
/// check_reach_nodes
///
/// BFS from `target_node` in the sampled adjacency to find
/// newly activated nodes up to `max_hop` steps.
//////////////////////////////////////////////////////////////
pub fn check_reach_nodes(
    target_node: usize,
    sampled_graph: &HashMap<usize, Vec<usize>>,
    curr_active_nodes: &HashSet<usize>,
    max_hop: Option<usize>,
) -> HashSet<usize> {
    let mut curr_active_nodes_copy = curr_active_nodes.clone();
    let mut newly_activated: HashSet<usize> = HashSet::new();
    newly_activated.insert(target_node);

    let max_steps = max_hop.unwrap_or(std::usize::MAX);
    let mut steps = 0;

    while !newly_activated.is_empty() && steps < max_steps {
        let mut next_b = HashSet::new();
        for &node in newly_activated.iter() {
            if let Some(neighbors) = sampled_graph.get(&node) {
                for &nbr in neighbors {
                    if !curr_active_nodes_copy.contains(&nbr) {
                        next_b.insert(nbr);
                    }
                }
            }
        }
        for &x in &next_b {
            curr_active_nodes_copy.insert(x);
        }
        newly_activated = next_b;
        steps += 1;
    }

    curr_active_nodes_copy
}

//////////////////////////////////////////////////////////////
/// single_perm_sample
///
/// Single-threaded version corresponding to `_single_perm_sample`.
/// It updates `delta_est` in-place.
//////////////////////////////////////////////////////////////
fn single_perm_sample(
    graph: &Graph,
    delta_est: &mut HashMap<usize, f64>,
    gen_seed: u64,
    max_hop: Option<usize>,
) -> Result<(), String> {
    // 1) Basic checks
    sanity_check(&graph.g, graph.seeds.as_ref())?;

    // 2) Prepare seeds
    let seeds_set = graph.seeds.as_ref().ok_or("No seeds in the graph")?;
    if seeds_set.is_empty() {
        return Ok(());
    }
    let mut seeds: Vec<usize> = seeds_set.iter().copied().collect();

    // 3) Sample entire live-edge graph once
    let sampled_graph = sample_live_edge_graph(graph, Some(gen_seed));

    // 4) Shuffle seeds
    let mut rng = StdRng::seed_from_u64(gen_seed);
    seeds.shuffle(&mut rng);

    // 5) Start with all seeds active
    let mut curr_active_nodes: HashSet<usize> = seeds_set.clone();

    // 6) For each seed in the permutation, find marginal contribution
    for &t_j in &seeds {
        let updated_set = check_reach_nodes(t_j, &sampled_graph, &curr_active_nodes, max_hop);
        let delta = updated_set.len() as f64 - curr_active_nodes.len() as f64;
        *delta_est.get_mut(&t_j).unwrap() += delta;
        curr_active_nodes = updated_set;
    }

    Ok(())
}

//////////////////////////////////////////////////////////////
/// single_perm_sample_parallel
///
/// Returns a local `delta_est` map for each permutation. This
/// is the parallel-friendly version of the sampling logic.
//////////////////////////////////////////////////////////////
fn single_perm_sample_parallel(
    graph: &Graph,
    gen_seed: u64,
    max_hop: Option<usize>,
) -> Result<HashMap<usize, f64>, String> {
    // 1) Check graph
    sanity_check(&graph.g, graph.seeds.as_ref())?;

    // 2) Prepare seeds
    let seeds_set = graph.seeds.as_ref().ok_or("No seeds in the graph")?;
    if seeds_set.is_empty() {
        let out = seeds_set.iter().map(|&s| (s, 0.0)).collect();
        return Ok(out);
    }
    let mut seeds: Vec<usize> = seeds_set.iter().copied().collect();
    let mut rng = StdRng::seed_from_u64(gen_seed);

    // 3) Sample entire live-edge graph
    let sampled_graph = sample_live_edge_graph(graph, Some(gen_seed));

    // 4) Shuffle seeds
    seeds.shuffle(&mut rng);

    // 5) Build local delta map
    let mut delta_est: HashMap<usize, f64> = seeds_set
        .iter()
        .map(|&s| (s, 0.0))
        .collect();

    // 6) BFS for each seed in perm
    let mut curr_active_nodes: HashSet<usize> = seeds_set.clone();
    for &t_j in &seeds {
        let updated_set = check_reach_nodes(t_j, &sampled_graph, &curr_active_nodes, max_hop);
        let delta = updated_set.len() as f64 - curr_active_nodes.len() as f64;
        *delta_est.get_mut(&t_j).unwrap() += delta;
        curr_active_nodes = updated_set;
    }

    Ok(delta_est)
}

//////////////////////////////////////////////////////////////
/// approx_permutation2
///
/// Implements the user's `approx_permutation2(...)` function,
/// now with a `parallel: bool` parameter. If `parallel` is true,
/// it uses Rayon to parallelize each permutation sampling; if false,
/// it runs single-threaded.
//////////////////////////////////////////////////////////////
pub fn approx_permutation2(
    graph: &Graph,
    gen_seed: Option<u64>,
    max_hop: Option<usize>,
    mut n: Option<usize>,
    epsilon: Option<f64>,
    delta: Option<f64>,
    parallel: bool,
) -> Result<HashMap<usize, f64>, String> {
    // 1) Basic checks
    sanity_check(&graph.g, graph.seeds.as_ref())?;

    // Configure Rayon to use at most 48 threads if running in parallel mode
    if parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(48)
            .build_global()
            .unwrap_or_else(|e| {
                eprintln!("Warning: Failed to configure thread pool: {}", e);
            });
    }

    let seeds_set = graph.seeds.as_ref().ok_or("No seeds in the graph")?;
    let n_seeds = seeds_set.len();
    if n_seeds == 0 {
        return Ok(HashMap::new());
    }

    // 2) If n is None, compute it from the formula in your Python code
    if n.is_none() {
        let n_vertices = graph.n;
        let eps = epsilon.ok_or("epsilon not provided")?;
        let dlt = delta.ok_or("delta not provided")?;

        // n = ceil((n_vertices^2 / (2 * eps^2)) * ln(2 * n_seeds / dlt))
        let inside_log = 2.0 * n_seeds as f64 / dlt;
        let log_part = inside_log.ln().max(0.0);
        let val = (n_vertices.pow(2) as f64 / (2.0 * eps.powi(2))) * log_part;
        n = Some(val.ceil() as usize);
    }
    let n_val = n.unwrap();
    println!("parameters: n={}", n_val);

    // 3) Initialize counters
    let base_seed = gen_seed.unwrap_or(0);
    let mut delta_est: HashMap<usize, f64> = seeds_set
        .iter()
        .map(|&s| (s, 0.0))
        .collect();

    // 4) Run the permutations in parallel or single-thread
    if parallel {
        println!("Parallel mode with rayon...");
        // Generate partial results in parallel
        let partials: Vec<HashMap<usize, f64>> = (0..n_val)
            .into_par_iter()
            .map(|i| {
                let seed_i = base_seed.wrapping_add(i as u64);
                // each iteration calls single_perm_sample_parallel
                single_perm_sample_parallel(graph, seed_i, max_hop)
                    .unwrap_or_else(|_| {
                        // If an error occurs, return zero map
                        seeds_set.iter().map(|&s| (s, 0.0)).collect()
                    })
            })
            .collect();

        // Combine partials
        for local_est in partials {
            for (&node, &val) in local_est.iter() {
                *delta_est.get_mut(&node).unwrap() += val;
            }
        }
    } else {
        println!("Single-threaded mode...");
        for i in 0..n_val {
            let seed_i = base_seed.wrapping_add(i as u64);
            single_perm_sample(graph, &mut delta_est, seed_i, max_hop)?;
        }
    }

    // 5) Average
    let mut final_est = HashMap::new();
    for &t in seeds_set {
        final_est.insert(t, delta_est[&t] / n_val as f64);
    }

    Ok(final_est)
}