use crate::graph_utils::{Graph, sanity_check};
use std::collections::{HashMap, HashSet, VecDeque};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;

/// Builds a modified and reversed version of the graph
pub fn build_modified_and_reverse_graph(original: &Graph) -> Graph {
    let mut rev_graph = Graph::new();
    rev_graph.n = original.n;
    rev_graph.seeds = original.seeds.clone();
    // skip edges that lead into seeds, reverse the rest
    for edge_ref in original.g.edge_references() {
        let u = edge_ref.source().index();
        let v = edge_ref.target().index();
        let p_uv = *edge_ref.weight();
        if let Some(seeds) = &original.seeds {
            if seeds.contains(&v) {
                continue;
            }
        }
        rev_graph.add_edge(v, u, p_uv);
    }
    rev_graph
}

/// Generate one random RR set in the reversed graph
pub fn generate_rr_set(
    rev_graph: &Graph, 
    non_seeds: &[usize], 
    rng: &mut StdRng,
    max_hop: Option<usize>
) -> HashSet<usize> {
    if non_seeds.is_empty() {
        return HashSet::new();
    }
    
    let start_index = rng.gen_range(0..non_seeds.len());
    let start_node = non_seeds[start_index];
    
    let mut visited = HashSet::new();
    visited.insert(start_node);
    
    let mut frontier = HashSet::new();
    frontier.insert(start_node);
    
    let mut steps = 0;
    let max_hop = max_hop.unwrap_or(usize::MAX);
    
    while !frontier.is_empty() && steps < max_hop {
        let mut next_frontier = HashSet::new();
        
        for &curr in &frontier {
            if let Some(curr_idx) = rev_graph.node_map.get(&curr) {
                for edge_ref in rev_graph.g.edges(*curr_idx) {
                    let neighbor_idx = edge_ref.target();
                    let neighbor = neighbor_idx.index();
                    if !visited.contains(&neighbor) {
                        let p_edge = rev_graph.get_edge_probability(curr, neighbor);
                        if rng.gen::<f64>() <= p_edge {
                            visited.insert(neighbor);
                            next_frontier.insert(neighbor);
                        }
                    }
                }
            }
        }
        
        frontier = next_frontier;
        steps += 1;
    }
    
    visited
}

/// Approximates Shapley values using the RR-set approach.
/// 
/// If `num_theta` is Some(t), we skip Phase 1 and sample exactly t RR sets in Phase 2.
/// If `parallel` is true, we parallelize the RR-set generation in both phases.
/// If `max_hop` is specified, RR-sets will only include nodes within that many hops from the starting node.
pub fn rr_shapley_approx(
    graph: &Graph,
    k: usize,
    eps: f64,
    ell: f64,
    random_seed: Option<u64>,
    num_theta: Option<usize>,
    parallel: bool,
    max_hop: Option<usize>,
) -> HashMap<usize, f64> {
    // Configure Rayon to use at most 48 threads if running in parallel mode
    if parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(48)
            .build_global()
            .unwrap_or_else(|e| {
                eprintln!("Warning: Failed to configure thread pool: {}", e);
            });
    }

    // Basic checks
    let _ = sanity_check(&graph.g, graph.seeds.as_ref());
    let seed_set = match &graph.seeds {
        Some(s) if !s.is_empty() => s.clone(),
        _ => return HashMap::new(),
    };
    // Reverse graph
    let rev_graph = build_modified_and_reverse_graph(graph);
    // Collect non-seeds
    let all_nodes: HashSet<usize> = rev_graph.g.node_indices().map(|idx| idx.index()).collect();
    let non_seeds: Vec<usize> = all_nodes.difference(&seed_set).copied().collect();
    let n_prime = non_seeds.len();
    if n_prime == 0 {
        return seed_set.iter().map(|&t| (t, 0.0)).collect();
    }
    // RNG
    let base_rng_seed = random_seed.unwrap_or(0);
    
    // We define a helper to generate (theta_count) RR sets and update partial counters
    // in single or parallel fashion:
    fn sample_rr_sets(
        rev_graph: &Graph,
        non_seeds: &[usize],
        seed_set: &HashSet<usize>,
        start_seed: u64,
        rr_count: usize,
        parallel: bool,
        max_hop: Option<usize>,
        existing_est: &mut HashMap<usize, f64>,
    ) {
        if parallel {
            // Parallel: chunk [0..rr_count) into a par_iter, each iteration uses its own RNG
            let partials: Vec<HashMap<usize, f64>> = (0..rr_count)
                .into_par_iter()
                .map(|i| {
                    let mut local_rng = StdRng::seed_from_u64(start_seed.wrapping_add(i as u64));
                    let rr_set = generate_rr_set(rev_graph, non_seeds, &mut local_rng, max_hop);
                    let intersect: Vec<_> = rr_set.intersection(seed_set).copied().collect();
                    if intersect.is_empty() {
                        // Return an empty partial
                        let local_map: HashMap<usize, f64> = HashMap::new();
                        return local_map;
                    } else {
                        let mut local_map: HashMap<usize, f64> = HashMap::new();
                        let c_int = intersect.len();
                        let contrib = 1.0 / (c_int as f64);
                        for &t in &intersect {
                            local_map.insert(t, contrib);
                        }
                        local_map
                    }
                })
                .collect();

            // Merge partials
            for part_map in partials {
                for (node, val) in part_map {
                    *existing_est.get_mut(&node).unwrap() += val;
                }
            }
        } else {
            // Single-thread: a loop
            let mut rng = StdRng::seed_from_u64(start_seed);
            for i in 0..rr_count {
                // You can create a new RNG each iteration or just advance the single rng
                let rr_set = generate_rr_set(rev_graph, non_seeds, &mut rng, max_hop);
                let intersect: Vec<_> = rr_set.intersection(seed_set).copied().collect();
                if !intersect.is_empty() {
                    let c_int = intersect.len();
                    let contrib = 1.0 / c_int as f64;
                    for &t in &intersect {
                        *existing_est.get_mut(&t).unwrap() += contrib;
                    }
                }
            }
        }
    }

    let mut lb = 1.0;
    // Phase 1 if num_theta is None
    if num_theta.is_none() {
        let eps_prime = (2.0_f64).sqrt() * eps;
        let mut est = HashMap::new();
        for &s in &seed_set {
            est.insert(s, 0.0);
        }
        let mut theta0 = 0usize;
        let max_i = if n_prime > 1 {
            (n_prime as f64).log2().floor() as usize
        } else {
            0
        };
        for i in 1..max_i {
            let x_i = n_prime as f64 / 2.0_f64.powi(i as i32);
            // eqn(9)-like
            let ln_term = {
                let n_prime_f = n_prime as f64;
                let seed_count_f = seed_set.len() as f64;
                let log2n = if n_prime > 1 { n_prime_f.log2() } else { 1.0 };
                ell * n_prime_f.ln() + seed_count_f.ln() + log2n.ln() + 2.0_f64.ln()
            };
            let eps_prime_sq = eps_prime * eps_prime;
            let frac_term = 2.0 + (2.0 / 3.0)*eps_prime;
            let x_i_factor = n_prime as f64 * frac_term / (eps_prime_sq * x_i);
            let raw_theta_i = x_i_factor * ln_term;
            let theta_i = raw_theta_i.ceil() as usize;
            let needed = theta_i.saturating_sub(theta0);

            // sample needed RR sets
            sample_rr_sets(&rev_graph, &non_seeds, &seed_set, 
                           base_rng_seed.wrapping_add(i as u64), needed, parallel, max_hop, &mut est);
            theta0 = theta_i;

            // pick k-th largest
            let mut vals: Vec<_> = est.values().copied().collect();
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if k == 0 || k > vals.len() {
                break;
            }
            let est_k = vals[k-1];
            let shap_k = (n_prime as f64)* est_k / (theta_i as f64);
            if shap_k >= (1.0 + eps_prime)* x_i {
                lb = (n_prime as f64 * est_k)/(theta_i as f64*(1.0 + eps_prime));
                break;
            }
        }
    } 
    // else skip phase 1, lb stays =1.0

    // Phase 2
    let final_theta = if let Some(t) = num_theta {
        t
    } else {
        // eqn(23):
        let frac_term_final = 2.0 + (2.0/3.0)*eps;
        let ln4 = (4.0_f64).ln();
        let n_prime_f = n_prime as f64;
        let seed_count_f = seed_set.len() as f64;
        let ln_term_2 = ell*n_prime_f.ln() + seed_count_f.ln() + ln4;
        let raw_theta = (n_prime_f * frac_term_final* ln_term_2)/(eps*eps * lb);
        raw_theta.ceil() as usize
    };
    
    // create a fresh est for phase 2
    let mut est2 = HashMap::new();
    for &s in &seed_set {
        est2.insert(s, 0.0);
    }

    // sample final_theta times
    sample_rr_sets(
        &rev_graph,
        &non_seeds,
        &seed_set,
        base_rng_seed.wrapping_add(999999), // shift the seed, or pick any offset
        final_theta,
        parallel,
        max_hop,
        &mut est2
    );

    // shap(t) = n_prime * est2[t] / final_theta
    let mut shap_approx = HashMap::new();
    let n_prime_f = n_prime as f64;
    for &t in &seed_set {
        let val = *est2.get(&t).unwrap_or(&0.0);
        shap_approx.insert(t, (n_prime_f*val)/(final_theta as f64));
    }

    shap_approx
}