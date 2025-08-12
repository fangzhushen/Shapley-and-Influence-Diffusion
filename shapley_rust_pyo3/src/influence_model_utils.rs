use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand::rngs::ThreadRng;
use crate::graph_utils::Graph;
use petgraph::graph::{DiGraph, NodeIndex};
use statrs::function::factorial::ln_factorial;

/// Computes coefficients for a given number of seeds.
///
/// Returns a vector of coefficients, not their logarithms.
///
/// # Arguments
/// * `n_seeds` - Number of seeds.
///
/// # Returns
/// * `Ok(Vec<f64>)` on success.
/// * `Err(String)` on overflow.
// pub fn compute_coefficients_log(n_seeds: usize) -> Result<Vec<f64>, String> {
//     let mut c = Vec::new();
//     for k in 0..n_seeds {
//         let log_val = if k == 0 || k == n_seeds - 1 {
//             (1.0 / n_seeds as f64).ln()
//         } else {
//             ln_factorial(k as u64) + ln_factorial((n_seeds - k) as u64) - ln_factorial(n_seeds as u64)
//         };
//         let val = log_val.exp();
//         if val.is_infinite() || val.is_nan() {
//             return Err(format!("Overflow in compute_coefficients_log for k={}", k));
//         }
//         c.push(val);
//     }
//     Ok(c)
// }

pub fn compute_coefficients_log(n_seeds: usize) -> Result<Vec<f64>, String> {
    let mut c = Vec::new();

    for k in 0..n_seeds {
        let log_val = if k == 0 || k == n_seeds - 1 {
            (1.0 / n_seeds as f64).ln()
        } else {
            // Matches Python:
            // log_val = ln(k!) + ln((n_seeds - k - 1)!) - ln(n_seeds!)
            ln_factorial(k as u64)
                + ln_factorial((n_seeds - k - 1) as u64)
                - ln_factorial(n_seeds as u64)
        };

        let val = log_val.exp();
        if val.is_infinite() || val.is_nan() {
            return Err(format!("Overflow in compute_coefficients_log for k={}", k));
        }
        c.push(val);
    }

    Ok(c)
}

/// Counts the number of incoming edges to a node.
///
/// # Arguments
/// * `g` - Adjacency list from Graph.
/// * `node` - Node to count in-degree for.
///
/// # Returns
/// Number of incoming edges.
pub fn in_degree(g: &HashMap<usize, Vec<usize>>, node: usize) -> usize {
    g.iter()
        .filter(|(_, neighbors)| neighbors.contains(&node))
        .count()
}

/// Counts the number of outgoing edges from a node.
///
/// # Arguments
/// * `g` - Adjacency list from Graph.
/// * `node` - Node to count out-degree for.
///
/// # Returns
/// Number of outgoing edges.
pub fn out_degree(g: &HashMap<usize, Vec<usize>>, node: usize) -> usize {
    g.get(&node).map_or(0, |neighbors| neighbors.len())
}

/// Simulates the Independent Cascade (IC) model.
///
/// Returns the number of newly activated nodes (excluding seeds).
///
/// # Arguments
/// * `graph` - Graph instance.
/// * `s` - Optional initial seed set; uses graph.seeds if None.
/// * `rng` - Optional random number generator.
/// * `max_hop` - Optional maximum number of hops; unlimited if None.
///
/// # Returns
/// Number of newly activated nodes.
pub fn ic_model(
    graph: &Graph,
    s: Option<&HashSet<usize>>,
    max_hop: Option<usize>,
) -> usize {
    let binding = HashSet::new();
    let s = s.unwrap_or(graph.seeds.as_ref().unwrap_or(&binding));
    if s.is_empty() {
        return 0;
    }

    let mut rng = rand::thread_rng();
    let mut activated = graph.seeds.as_ref().unwrap_or(&HashSet::new()).clone();
    let mut frontier: HashSet<usize> = s.clone();
    
    let mut steps = 0;
    let max_hop = max_hop.unwrap_or(usize::MAX);

    while !frontier.is_empty() && steps < max_hop {
        let mut next_frontier = HashSet::new();
        
        for &node in &frontier {
            let node_idx = NodeIndex::new(node);
            for neighbor in graph.g.neighbors(node_idx) {
                let target = neighbor.index();
                if !activated.contains(&target) {
                    let probability = graph.p.as_ref()
                        .and_then(|p| p.get(&(node, target)))
                        .unwrap_or(&0.0);
                    if rng.gen::<f64>() <= *probability {
                        next_frontier.insert(target);
                        activated.insert(target);
                    }
                }
            }
        }
        
        frontier = next_frontier;
        steps += 1;
    }

    activated.len() - graph.seeds.as_ref().map_or(0, |seeds| seeds.len())
}

/// Evaluates influence spread using Monte Carlo simulation.
///
/// Returns the mean and standard deviation of the spread.
///
/// # Arguments
/// * `graph` - Graph instance.
/// * `s` - Seed set.
/// * `rng` - Optional random number generator.
/// * `max_hop` - Optional maximum number of hops.
/// * `no_simulations` - Number of simulation runs (default 100).
///
/// # Returns
/// Tuple of (mean, std).
pub fn monte_carlo_simulation(
    graph: &Graph,
    s: &HashSet<usize>,
    max_hop: Option<usize>,
    no_simulations: usize,
) -> (f64, f64) {
    // let mut rng = rng.unwrap_or_else(rand::thread_rng);
    let mut results = Vec::with_capacity(no_simulations);
    for _ in 0..no_simulations {
        results.push(ic_model(graph, Some(s), max_hop));
    }
    let mean_val = results.iter().sum::<usize>() as f64 / no_simulations as f64;
    let variance = results.iter()
        .map(|&x| (x as f64 - mean_val).powi(2))
        .sum::<f64>() / no_simulations as f64;
    let std_val = variance.sqrt();
    (mean_val, std_val)
}

/// Computes the exact expected influence spread under the IC model.
///
/// Uses brute force over all edge subsets.
///
/// # Arguments
/// * `graph` - Graph instance.
/// * `s` - Optional seed set; uses graph.seeds if None.
///
/// # Returns
/// Expected number of activated nodes.
pub fn exact_value_function(graph: &Graph, s: Option<&HashSet<usize>>) -> f64 {
    let binding = HashSet::new();
    let s = s.unwrap_or(graph.seeds.as_ref().unwrap_or(&binding));
    if s.is_empty() {
        return 0.0;
    }

    let mut edges_list = Vec::new();
    for edge in graph.g.edge_indices() {
        let (u, v) = graph.g.edge_endpoints(edge).unwrap();
        edges_list.push((u.index(), v.index()));
    }
    let num_edge = edges_list.len();
    let mut expected_spread = 0.0;

    for mask in 0..(1 << num_edge) {
        let mut subset_probability = 1.0;
        for i in 0..num_edge {
            let (u, v) = edges_list[i];
            let p_uv = graph.p.as_ref().and_then(|p| p.get(&(u, v))).unwrap_or(&0.0);
            if (mask & (1 << i)) != 0 {
                subset_probability *= p_uv;
            } else {
                subset_probability *= 1.0 - p_uv;
            }
            if subset_probability == 0.0 {
                break;
            }
        }
        if subset_probability == 0.0 {
            continue;
        }
        let mut activated = s.clone();
        let mut queue: Vec<usize> = s.iter().copied().collect();
        while let Some(current) = queue.pop() {
            for (i, &(u, v)) in edges_list.iter().enumerate() {
                if u == current && (mask & (1 << i)) != 0 && !activated.contains(&v) {
                    activated.insert(v);
                    queue.push(v);
                }
            }
        }
        expected_spread += activated.len() as f64 * subset_probability;
    }
    expected_spread
}