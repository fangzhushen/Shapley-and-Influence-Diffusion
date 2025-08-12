use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use std::collections::{HashMap, HashSet};

mod graph_utils;
mod influence_model_utils;
mod permutation_shapley;
mod permutation2_shapley;
mod rr_shapley;
mod single_shapley;
mod single_shapley_optimize;
use crate::graph_utils::Graph;
use single_shapley::{single_step_shapley, exact_single_step_shapley};

#[pyclass]
struct PyGraph {
    inner: Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        PyGraph {
            inner: Graph::new()
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, probability: f64) {
        self.inner.add_edge(from, to, probability);
    }

    fn add_seeds(&mut self, seeds: HashSet<usize>) -> PyResult<(String, usize)> {
        self.inner.add_seeds(seeds)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    fn select_seeds_random(&mut self, k: usize, gen_seed: Option<u64>) -> PyResult<(String, usize)> {
        self.inner.select_seeds_random(k, gen_seed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    fn select_seeds_degree(&mut self, k: usize) -> PyResult<(String, usize)> {
        self.inner.select_seeds_degree(k)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    fn add_uniform_prob(&mut self, p: f64) -> String {
        self.inner.add_uniform_prob(p)
    }

    fn add_wc_prob(&mut self) -> String {
        self.inner.add_wc_prob()
    }

    fn save_to_file(&self, filepath: &str) -> PyResult<()> {
        self.inner.save_to_file(filepath)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
    }

    #[staticmethod]
    fn load_from_file(filepath: &str) -> PyResult<Self> {
        Graph::load_from_file(filepath)
            .map(|g| PyGraph { inner: g })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn get_seeds(&self) -> Option<HashSet<usize>> {
        self.inner.get_seeds()
    }

    fn get_nodes(&self) -> HashSet<usize> {
        self.inner.get_nodes()
    }

    fn get_edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.get_edges()
    }
}

#[pyfunction]
fn read_undirect_unweight_graph(filename: &str) -> PyResult<PyGraph> {
    graph_utils::read_undirect_unweight_graph(filename)
        .map(|g| PyGraph { inner: g })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

#[pyfunction]
fn read_direct_unweighted_graph(filename: &str) -> PyResult<PyGraph> {
    graph_utils::read_direct_unweighted_graph(filename)
        .map(|g| PyGraph { inner: g })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

#[pyfunction]
fn read_direct_weighted_graph(filename: &str) -> PyResult<PyGraph> {
    graph_utils::read_direct_weighted_graph(filename)
        .map(|g| PyGraph { inner: g })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

#[pyfunction]
fn read_networkx_edgelist(filename: &str) -> PyResult<PyGraph> {
    graph_utils::read_networkx_edgelist(filename)
        .map(|g| PyGraph { inner: g })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

#[pyfunction]
fn approx_permutation1(
    graph: &PyGraph,
    parallel: bool,
    gen_seed: Option<u64>,
    max_hop: Option<usize>,
    n: Option<usize>,
    m: Option<usize>,
    epsilon: Option<f64>,
    delta: Option<f64>,
    n_jobs: Option<usize>,
) -> HashMap<usize, f64> {
    permutation_shapley::approximate_shapley_permutation_parallel(
        &graph.inner,
        parallel,
        gen_seed,
        max_hop,
        n,
        m,
        epsilon,
        delta,
        n_jobs,
    )
}
    
#[pyfunction]
fn approx_permutation2(
    graph: &PyGraph,
    gen_seed: Option<u64>,
    max_hop: Option<usize>,
    n: Option<usize>,
    epsilon: Option<f64>,
    delta: Option<f64>,
    parallel: bool,
) -> PyResult<HashMap<usize, f64>> {
    permutation2_shapley::approx_permutation2(
        &graph.inner,
        gen_seed,
        max_hop,
        n,
        epsilon,
        delta,
        parallel,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pyfunction]
fn rr_shapley_approx(
    graph: &PyGraph,
    k: usize,
    eps: f64,
    ell: f64,
    random_seed: Option<u64>,
    num_theta: Option<usize>,
    parallel: bool,
    max_hop: Option<usize>,
) -> HashMap<usize, f64> {
    rr_shapley::rr_shapley_approx(
        &graph.inner,
        k,
        eps,
        ell,
        random_seed,
        num_theta,
        parallel,
        max_hop,
    )
}

#[pyfunction]
fn single_step_shapley_rust(
    graph: &PyGraph,
) -> PyResult<HashMap<usize, f64>> {
    single_shapley::single_step_shapley(&graph.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pyfunction]
fn exact_single_step_shapley_rust(
    graph: &PyGraph,
) -> PyResult<HashMap<usize, f64>> {
    single_shapley::exact_single_step_shapley(&graph.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pyfunction]
fn single_step_shapley_optimize_rust(
    graph: &PyGraph,
) -> PyResult<HashMap<usize, f64>> {
    single_shapley_optimize::single_step_shapley_optimize(&graph.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pymodule]
fn shapley_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(read_undirect_unweight_graph, m)?)?;
    m.add_function(wrap_pyfunction!(read_direct_unweighted_graph, m)?)?;
    m.add_function(wrap_pyfunction!(read_direct_weighted_graph, m)?)?;
    m.add_function(wrap_pyfunction!(read_networkx_edgelist, m)?)?;
    m.add_function(wrap_pyfunction!(approx_permutation1, m)?)?;
    m.add_function(wrap_pyfunction!(approx_permutation2, m)?)?;
    m.add_function(wrap_pyfunction!(rr_shapley_approx, m)?)?;
    m.add_function(wrap_pyfunction!(single_step_shapley_rust, m)?)?;
    m.add_function(wrap_pyfunction!(exact_single_step_shapley_rust, m)?)?;
    m.add_function(wrap_pyfunction!(single_step_shapley_optimize_rust, m)?)?;
    Ok(())
} 