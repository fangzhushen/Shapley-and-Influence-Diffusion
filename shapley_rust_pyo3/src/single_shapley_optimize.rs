use std::collections::{HashMap, HashSet, BTreeMap};
use crate::graph_utils::{Graph, sanity_check};
use crate::influence_model_utils::{compute_coefficients_log};
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use statrs::function::factorial::ln_factorial;
use itertools::Itertools;
use std::f64;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;


// /// Computes n choose k using a more numerically stable method
// fn combination(n: usize, k: usize) -> f64 {
//     if k > n {
//         return 0.0;
//     }
//     if k == 0 || k == n {
//         return 1.0;
//     }
    
//     // Use smaller k to reduce computation
//     let k = k.min(n - k);
    
//     let mut result = 1.0;
//     for i in 0..k {
//         result *= (n - i) as f64;
//         result /= (i + 1) as f64;
//     }
    
//     result
// }

fn combination(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    
    let k = k.min(n - k);
    
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    
    result
}


// fn compute_alpha_values_local(
//     t: usize,
//     u: usize,
//     local_seeds: &[usize],
//     get_prob: &impl Fn(usize, usize) -> f64,
// ) -> Vec<f64> {
//     let n = local_seeds.len();
//     // alpha[k] will represent the probability that exactly k of the local seeds (other than t) fail to activate u.
//     let mut alpha = vec![0.0; n];
//     alpha[0] = 1.0;

//     let mut l = 0;
//     for &s in local_seeds {
//         if s == t {
//             continue;
//         }
//         let p_su = get_prob(s, u);
//         l += 1;
//         alpha[l] = alpha[l - 1] * (1.0 - p_su);

//         if l >= 2 {
//             // Update alpha[k] backward to account for an additional potential seed s.
//             for k in (1..l).rev() {
//                 alpha[k] = alpha[k - 1] * (1.0 - p_su) + alpha[k];
//             }
//         }
//     }

//     alpha
// }

fn compute_alpha_values_local(
    t: usize,
    u: usize,
    local_seeds: &[usize],
    get_prob: &impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let n = local_seeds.len();
    let mut alpha = vec![0.0; n];
    alpha[0] = 1.0;

    let mut l = 0;
    for &s in local_seeds {
        if s == t {
            continue;
        }
        let p_su = get_prob(s, u);
        l += 1;
        alpha[l] = alpha[l - 1] * (1.0 - p_su);

        if l >= 2 {
            for k in (1..l).rev() {
                alpha[k] = alpha[k - 1] * (1.0 - p_su) + alpha[k];
            }
        }
    }

    alpha
}

pub fn single_step_shapley_optimize(graph: &Graph) -> Result<HashMap<usize, f64>, String> {
    sanity_check(&graph.g, graph.seeds.as_ref())?;

    // Configure Rayon to use 48 threads
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(48)
        .build()
        .map_err(|e| format!("Failed to build thread pool: {}", e))?;

    // The rest of the function will run inside this thread pool
    thread_pool.install(|| {
        let seeds = match &graph.seeds {
            Some(s) => s,
            None => return Err("No seed nodes provided".to_string()),
        };
        if seeds.is_empty() {
            return Err("Empty seed set provided".to_string());
        }

        let mut shapley_values: HashMap<usize, f64> = seeds.iter().map(|&s| (s, 0.0)).collect();

        let get_prob = |from: usize, to: usize| -> f64 {
            graph
                .p
                .as_ref()
                .and_then(|p| p.get(&(from, to)))
                .copied()
                .unwrap_or(0.0)
        };

        let mut non_seed_neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
        for &seed in seeds {
            for neighbor in graph.g.neighbors(NodeIndex::new(seed)) {
                let u = neighbor.index();
                if !seeds.contains(&u) {
                    non_seed_neighbors.entry(u).or_default().push(seed);
                }
            }
        }

        // Convert to a vector so we can use .par_iter()
        let non_seed_vec: Vec<(usize, Vec<usize>)> = non_seed_neighbors.into_iter().collect();

        // Process in parallel
        let partial_results: Vec<HashMap<usize, f64>> = non_seed_vec
            .par_iter()
            .map(|(u, incoming_seeds)| {
                let mut local_map = HashMap::new();
                if incoming_seeds.is_empty() {
                    return local_map;
                }

                let n_local = incoming_seeds.len();
                let c_local = match compute_coefficients_log(n_local) {
                    Ok(coeffs) => coeffs,
                    Err(_) => return local_map,
                };

                for &t in incoming_seeds {
                    let p_tu = get_prob(t, *u);
                    if p_tu == 0.0 {
                        continue;
                    }
                    let alpha_local = compute_alpha_values_local(t, *u, &incoming_seeds, &get_prob);

                    for k in 0..n_local {
                        *local_map.entry(t).or_insert(0.0) += p_tu * alpha_local[k] * c_local[k];
                    }
                }

                local_map
            })
            .collect();

        // Merge partial results back into the global shapley_values
        for local_map in partial_results {
            for (seed, val) in local_map {
                *shapley_values.entry(seed).or_insert(0.0) += val;
            }
        }

        Ok(shapley_values)
    })
}



/// Update alpha values for a local subgraph
fn update_alpha_values_local(
    t: usize,
    u: usize,
    alpha_prev: &[f64],
    last_t: usize,
    n_local_seeds: usize,
    get_prob: &impl Fn(usize, usize) -> f64,
    local_seeds: &[usize]
) -> Vec<f64> {
    let mut alpha = alpha_prev.to_vec();
    let p_tprime_u = get_prob(last_t, u);
    let p_tu = get_prob(t, u);
    let l = n_local_seeds - 1;
    
    // Find position of t and last_t in the local_seeds array
    let t_pos = local_seeds.iter().position(|&s| s == t).unwrap_or(0);
    let last_t_pos = local_seeds.iter().position(|&s| s == last_t).unwrap_or(0);
    
    // Remove old seed influence
    for k in 1..=l {
        alpha[k] = alpha[k] - alpha[k - 1] * (1.0 - p_tu);
    }
    
    // Add new seed influence
    alpha[l] = alpha[l - 1] * (1.0 - p_tprime_u);
    
    if l >= 2 {
        for k in (1..l).rev() {
            alpha[k] = alpha[k - 1] * (1.0 - p_tprime_u) + alpha[k];
        }
    }
    
    alpha
}

/// Single-step dynamic programming algorithm for Shapley values
pub fn single_step_shapley(graph: &Graph) -> Result<HashMap<usize, f64>, String> {
    // Validate input
    sanity_check(&graph.g, graph.seeds.as_ref())?;
    
    // Check if seeds exist
    let seeds = match &graph.seeds {
        Some(s) => s,
        None => return Err("No seed nodes provided".to_string()),
    };
    
    if seeds.is_empty() {
        return Err("Empty seed set provided".to_string());
    }
    
    let n_seeds = seeds.len();
    let seeds_vec: Vec<usize> = seeds.iter().copied().collect(); 
    
    // Precompute coefficients
    let c = compute_coefficients_log(n_seeds)?;
    
    // Track alpha vectors and last computed seed
    let mut node_alpha: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut last_computed_for: HashMap<usize, usize> = HashMap::new();
    
    // Initialize Shapley values
    let mut shap: HashMap<usize, f64> = seeds.iter().map(|&seed| (seed, 0.0)).collect();
    
    // Helper closure to get edge probability
    let get_prob = |from: usize, to: usize| -> f64 {
        graph.p.as_ref()
            .and_then(|p| p.get(&(from, to)))
            .copied()
            .unwrap_or(0.0)
    };
    
    // For each seed
    for &t in &seeds_vec {
        // Find non-seed neighbors of t
        let neighbors: Vec<usize> = graph.g.neighbors(NodeIndex::new(t))
            .map(|n| n.index())
            .filter(|&u| !seeds.contains(&u))
            .collect();
        
        if !neighbors.is_empty() {
            for &u in &neighbors {
                let p_tu = get_prob(t, u);
                
                // Compute alpha values for node u with respect to seed t
                let alpha = compute_alpha_values(t, u, &seeds_vec, n_seeds, 
                                               &mut node_alpha, &mut last_computed_for, 
                                               get_prob);
                
                if p_tu > 0.0 {
                    // Update Shapley value for seed t
                    for k in 0..n_seeds {
                        *shap.entry(t).or_insert(0.0) += p_tu * alpha[k] * c[k];
                    }
                }
            }
        }
    }
    
    Ok(shap)
}

/// Compute first alpha values for a node
fn compute_first_alpha_values(
    t: usize,
    u: usize,
    seeds: &[usize], 
    n_seeds: usize,
    get_prob: impl Fn(usize, usize) -> f64
) -> Vec<f64> {
    let mut alpha = vec![0.0; n_seeds];
    alpha[0] = 1.0;
    let mut l = 0;
    
    for &s in seeds {
        if s == t {
            continue;
        }
        
        l += 1;
        let p_su = get_prob(s, u);
        alpha[l] = alpha[l - 1] * (1.0 - p_su);
        
        if l >= 2 {
            for k in (1..l).rev() {
                alpha[k] = alpha[k - 1] * (1.0 - p_su) + alpha[k];
            }
        }
    }
    
    alpha
}

/// Update alpha values
fn update_alpha_values(
    t: usize,
    u: usize,
    alpha_prev: &[f64],
    last_t: usize,
    n_seeds: usize,
    get_prob: impl Fn(usize, usize) -> f64
) -> Vec<f64> {
    let mut alpha = alpha_prev.to_vec();
    let p_tprime_u = get_prob(last_t, u);
    let p_tu = get_prob(t, u);
    let l = n_seeds - 1;
    
    // Remove old seed influence
    for k in 1..=l {
        alpha[k] = alpha[k] - alpha[k - 1] * (1.0 - p_tu);
    }
    
    // Add new seed influence
    alpha[l] = alpha[l - 1] * (1.0 - p_tprime_u);
    
    if l >= 2 {
        for k in (1..l).rev() {
            alpha[k] = alpha[k - 1] * (1.0 - p_tprime_u) + alpha[k];
        }
    }
    
    alpha
}

/// Helper to compute the alpha vector
fn compute_alpha_values(
    t: usize,
    u: usize,
    seeds: &[usize],
    n_seeds: usize,
    node_alpha: &mut HashMap<usize, Vec<f64>>,
    last_computed_for: &mut HashMap<usize, usize>,
    get_prob: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    // Check if we've computed alpha for this node before
    if !last_computed_for.contains_key(&u) {
        // First time computing alpha for node u
        let alpha = compute_first_alpha_values(t, u, seeds, n_seeds, &get_prob);
        last_computed_for.insert(u, t);
        node_alpha.insert(u, alpha.clone());
        alpha
    } else {
        // We've computed alpha for some other seed before
        let last_t = *last_computed_for.get(&u).unwrap();
        
        if last_t == t {
            // Already computed for the same seed t
            node_alpha.get(&u).unwrap().clone()
        } else {
            // Need to update from last_t to t
            let alpha_updated = update_alpha_values(
                t, u, 
                &node_alpha.get(&u).unwrap(), 
                last_t, n_seeds, &get_prob
            );
            
            last_computed_for.insert(u, t);
            node_alpha.insert(u, alpha_updated.clone());
            alpha_updated
        }
    }
}

/// Exact computation of single-step Shapley values
pub fn exact_single_step_shapley(graph: &Graph) -> Result<HashMap<usize, f64>, String> {
    // Validate input
    sanity_check(&graph.g, graph.seeds.as_ref())?;
    
    // Check if seeds exist
    let seeds = match &graph.seeds {
        Some(s) => s,
        None => return Err("No seed nodes provided".to_string()),
    };
    
    if seeds.is_empty() {
        return Err("Empty seed set provided".to_string());
    }
    
    // Convert seeds to vector for easier iteration
    let seeds_vec: Vec<usize> = seeds.iter().copied().collect();
    let n_seeds = seeds.len();
    
    // Precompute coefficients
    let c = compute_coefficients_log(n_seeds)?;
    
    // Gather all nodes in the graph (including those that might only be targets)
    let mut all_nodes = HashSet::new();
    
    // Add nodes from node_map
    for &node in graph.node_map.keys() {
        all_nodes.insert(node);
    }
    
    // Add targets from edges
    for edge in graph.g.edge_references() {
        all_nodes.insert(edge.target().index());
    }
    
    // Convert to sorted vec for consistent iteration
    let all_nodes: Vec<usize> = all_nodes.into_iter().collect();
    
    // Initialize Shapley values
    let mut shap: HashMap<usize, f64> = seeds.iter().map(|&seed| (seed, 0.0)).collect();
    
    // Helper function to compute value function for a seed set
    let value_function = |s_set: &HashSet<usize>| -> f64 {
        if s_set.is_empty() {
            return 0.0;
        }
        
        let mut value = 0.0;
        
        for &u in &all_nodes {
            if seeds.contains(&u) {
                continue; // Skip seed nodes
            }
            
            // Probability that u is activated by at least one seed in S_set
            let mut prob_noactive = 1.0;
            
            for &s in s_set {
                let prob = graph.p.as_ref()
                    .and_then(|p| p.get(&(s, u)))
                    .copied()
                    .unwrap_or(0.0);
                    
                prob_noactive *= (1.0 - prob);
            }
            
            let prob_active = 1.0 - prob_noactive;
            value += prob_active;
        }
        
        value
    };
    
    // For each seed t, calculate its Shapley value
    for &t in &seeds_vec {
        // The set of seeds excluding t
        let other_seeds: Vec<usize> = seeds_vec.iter()
            .filter(|&&s| s != t)
            .copied()
            .collect();
        
        // Iterate over all subsets T of other_seeds
        for k in 0..=other_seeds.len() {
            for subset in other_seeds.iter().combinations(k) {
                let mut t_set: HashSet<usize> = subset.into_iter().copied().collect();
                
                // Compute U(T)
                let ut = value_function(&t_set);
                
                // Compute U(T ∪ {t})
                t_set.insert(t);
                let ut_plus_t = value_function(&t_set);
                
                // Marginal contribution = U(T ∪ {t}) - U(T)
                let mc = ut_plus_t - ut;
                
                // Add weighted contribution to Shapley value
                *shap.entry(t).or_insert(0.0) += c[k] * mc;
            }
        }
    }
    
    Ok(shap)
}
