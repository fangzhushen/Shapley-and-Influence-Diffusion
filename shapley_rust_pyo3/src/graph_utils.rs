use std::collections::{HashMap, HashSet};
use std::fs::{read_to_string, write, File};
use std::io::{BufReader, BufWriter};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Serialize, Deserialize};
use log::warn;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::BTreeMap;
use std::io::{Read, Write};

// Required dependencies in Cargo.toml:
// rand = "0.8.5"
// serde = { version = "1.0", features = ["derive"] }
// bincode = "1.3"
// log = "0.4"

/// Represents a graph with adjacency list, edge probabilities, and seed nodes.
#[derive(Debug)]
pub struct Graph {
    pub g: DiGraph<(), f64>,  // nodes have no weight, edges have f64 weights for probabilities
    pub node_map: HashMap<usize, NodeIndex>,  // maps our node IDs to rustworkx NodeIndex
    pub seeds: Option<HashSet<usize>>,
    pub n: usize,
    pub p: Option<BTreeMap<(usize, usize), f64>>,  // Store edge probabilities separately for serialization
}

impl Graph {
    /// Creates a new Graph instance.
    pub fn new() -> Self {
        Graph {
            g: DiGraph::new(),
            node_map: HashMap::new(),
            seeds: None,
            n: 0,
            p: Some(BTreeMap::new()),
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, probability: f64) {
        // Update n to maintain the highest node index
        self.n = self.n.max(from + 1).max(to + 1);
        
        // Add nodes to node_map if they don't exist
        let from_idx = *self.node_map.entry(from)
            .or_insert_with(|| {
                let idx = NodeIndex::new(from);
                while self.g.node_count() <= from {
                    self.g.add_node(());
                }
                idx
            });
        
        let to_idx = *self.node_map.entry(to)
            .or_insert_with(|| {
                let idx = NodeIndex::new(to);
                while self.g.node_count() <= to {
                    self.g.add_node(());
                }
                idx
            });
        
        // Add edge with probability as weight
        self.g.add_edge(from_idx, to_idx, probability);
        
        // Store probability in the separate map for serialization
        if let Some(p) = &mut self.p {
            p.insert((from, to), probability);
        }
    }

    pub fn add_seeds(&mut self, seeds: HashSet<usize>) -> Result<(String, usize), String> {
        if seeds.is_empty() {
            return Err("Seeds are empty".to_string());
        }
        self.seeds = Some(seeds);
        let len = self.seeds.as_ref().unwrap().len();
        Ok(("Seeds added successfully".to_string(), len))
    }

    pub fn select_seeds_random(&mut self, k: usize, gen_seed: Option<u64>) -> Result<(String, usize), String> {
        if k >= self.g.node_count() {
            return Err("Number of seeds is greater than the number of nodes".to_string());
        }
        let mut rng = match gen_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let mut nodes: Vec<usize> = self.node_map.keys().copied().collect();
        nodes.shuffle(&mut rng);
        let seeds = nodes.into_iter().take(k).collect::<HashSet<usize>>();
        self.add_seeds(seeds)
    }

    pub fn select_seeds_degree(&mut self, k: usize) -> Result<(String, usize), String> {
        if k >= self.g.node_count() {
            return Err("Number of seeds is greater than the number of nodes".to_string());
        }
        
        // Use node_map which is now properly populated
        let mut out_degs_count: Vec<(usize, usize)> = self.node_map.iter()
            .map(|(&u, &idx)| (self.g.neighbors_directed(idx, petgraph::Direction::Outgoing).count(), u))
            .collect();
        
        out_degs_count.sort_by(|a, b| b.0.cmp(&a.0)); // Reverse sort by out-degree
        let seeds = out_degs_count.into_iter()
            .take(k)
            .map(|(_, u)| u)
            .collect::<HashSet<usize>>();
        
        self.add_seeds(seeds)
    }

    pub fn add_uniform_prob(&mut self, p: f64) -> String {
        // Validate probability value
        if p < 0.0 || p > 1.0 {
            return "Error: Probability must be between 0 and 1".to_string();
        }

        // Update all edge weights to p
        for edge in self.g.edge_weights_mut() {
            *edge = p;
        }

        "Uniform probability added successfully".to_string()
    }

    pub fn add_wc_prob(&mut self) -> String {
        // Calculate in-degrees for all nodes
        let mut in_degrees: HashMap<usize, usize> = HashMap::new();
        
        // Count in-degrees using edge references
        for edge_ref in self.g.edge_references() {
            let target = edge_ref.target().index();
            *in_degrees.entry(target).or_insert(0) += 1;
        }

        // Update edge probabilities in both the graph and probability map
        if let Some(p) = &mut self.p {
            let edges_to_update: Vec<_> = p.keys().cloned().collect();
            
            for (from, to) in edges_to_update {
                let prob = 1.0 / (*in_degrees.get(&to).unwrap_or(&1)) as f64;
                
                // Update the graph edge
                if let Some(edge_idx) = self.g.find_edge(
                    self.node_map[&from],
                    self.node_map[&to]
                ) {
                    self.g[edge_idx] = prob;
                }
                
                // Update the probability map
                p.insert((from, to), prob);
            }
        }

        "Weighted cascade probability added successfully".to_string()
    }

    pub fn save(&self, writer: impl Write) -> Result<(), String> {
        let serializable: SerializableGraph = self.into();
        bincode::serialize_into(writer, &serializable).map_err(|e| e.to_string())
    }

    pub fn load(reader: impl Read) -> Result<Self, String> {
        let serializable: SerializableGraph = bincode::deserialize_from(reader)
            .map_err(|e| e.to_string())?;
        Ok(Graph::from(serializable))
    }

    pub fn get_neighbors(&self, node: usize) -> Vec<usize> {
        if let Some(idx) = self.g.node_indices().nth(node) {
            self.g.neighbors(idx)
                .map(|n| n.index())
                .collect()
        } else {
            vec![]
        }
    }

    pub fn get_edge_probability(&self, from: usize, to: usize) -> f64 {
        self.p.as_ref()
            .and_then(|p| p.get(&(from, to)))
            .copied()
            .unwrap_or(0.0)
    }

    /// Save the graph to a file
    pub fn save_to_file(&self, filepath: &str) -> Result<(), String> {
        let file = File::create(filepath).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        self.save(writer)
    }

    /// Load the graph from a file
    pub fn load_from_file(filepath: &str) -> Result<Self, String> {
        let file = File::open(filepath).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        Self::load(reader)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.g.node_count()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.g.edge_count()
    }

    /// Get the current seeds as a HashSet
    pub fn get_seeds(&self) -> Option<HashSet<usize>> {
        self.seeds.clone()
    }

    /// Get all nodes as a HashSet
    pub fn get_nodes(&self) -> HashSet<usize> {
        self.node_map.keys().copied().collect()
    }

    /// Get all edges as Vec of (from, to, probability)
    pub fn get_edges(&self) -> Vec<(usize, usize, f64)> {
        self.g.edge_references()
            .map(|edge| (
                edge.source().index(),
                edge.target().index(),
                *edge.weight()
            ))
            .collect()
    }
}

pub fn read_undirect_unweight_graph(filename: &str) -> Result<Graph, String> {
    let mut g = Graph::new();
    let content = read_to_string(filename).map_err(|e| e.to_string())?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("Invalid line format".to_string());
        }
        let u: usize = parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let v: usize = parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        g.add_edge(u, v, 0.0);
        g.add_edge(v, u, 0.0);
    }
    Ok(g)
}

pub fn read_direct_unweighted_graph(filename: &str) -> Result<Graph, String> {
    let mut g = Graph::new();
    let content = read_to_string(filename).map_err(|e| e.to_string())?;
    
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!("Invalid edge format in line: '{}'", line));
        }
        let u: usize = parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let v: usize = parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        // For unweighted graphs, use a default weight of 0.0
        g.add_edge(u, v, 0.0);
    }
    Ok(g)
}

pub fn read_direct_weighted_graph(filename: &str) -> Result<Graph, String> {
    let mut g = Graph::new();
    let content = read_to_string(filename).map_err(|e| e.to_string())?;
    
    // Skip the first line and process the rest
    for line in content.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err("Invalid edge format".to_string());
        }
        let u: usize = parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let v: usize = parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let prob: f64 = parts[2].parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
        g.add_edge(u, v, prob);
    }
    Ok(g)
}

pub fn read_networkx_edgelist(filename: &str) -> Result<Graph, String> {
    let mut g = Graph::new();
    let content = read_to_string(filename).map_err(|e| e.to_string())?;
    
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        // Split the line into parts
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err("Invalid line format: expected 'node1 node2 {weight: value}'".to_string());
        }
        
        // Parse nodes
        let u: usize = parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let v: usize = parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        
        // Parse weight from dictionary-like string
        let weight_str = parts[2..].join(" ");
        let weight = if let Some(w) = weight_str
            .split("weight':")  // Split on 'weight':
            .nth(1)  // Take the part after 'weight':
            .and_then(|s| s.split("}").next())  // Take everything before the closing brace
            .and_then(|s| s.trim().parse::<f64>().ok())  // Parse as f64
        {
            w
        } else {
            return Err("Invalid weight format".to_string());
        };
        
        g.add_edge(u, v, weight);
    }
    Ok(g)
}

pub fn sanity_check(
    g: &DiGraph<(), f64>,
    seeds: Option<&HashSet<usize>>,
) -> Result<(), String> {
    if g.node_count() == 0 {
        warn!("G is empty. Shapley values will be trivial.");
    }

    match seeds {
        None => {
            warn!("No seed nodes provided. Shapley values = 0 for all.");
        }
        Some(seed_set) => {
            if seed_set.is_empty() {
                warn!("Empty seed set provided. Shapley values = 0 for all.");
            }
        }
    }
    Ok(())
}

// Implement a serializable version of the graph
#[derive(Serialize, Deserialize)]
pub struct SerializableGraph {
    pub edges: Vec<(usize, usize, f64)>,
    pub seeds: Option<HashSet<usize>>,
    pub n: usize,
}

impl From<&Graph> for SerializableGraph {
    fn from(graph: &Graph) -> Self {
        let edges = graph.g.edge_references()
            .map(|edge| (
                edge.source().index(),
                edge.target().index(),
                *edge.weight()
            ))
            .collect();
        
        SerializableGraph {
            edges,
            seeds: graph.seeds.clone(),
            n: graph.n,
        }
    }
}

impl From<SerializableGraph> for Graph {
    fn from(sg: SerializableGraph) -> Self {
        let mut graph = Graph::new();
        graph.n = sg.n;
        graph.seeds = sg.seeds;
        
        for (from, to, prob) in sg.edges {
            graph.add_edge(from, to, prob);
        }
        
        graph
    }
}