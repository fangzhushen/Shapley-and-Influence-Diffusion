# Shapley Values for Measuring Contributions in Influence Propagation in Networks

This repository contains the code for the research paper "Shapley Value for Measuring Contribution in Influence". It provides a high-performance Rust implementation of several Shapley value algorithms, exposed as a Python module for ease of use in experiments.

## Project Structure

The repository is organized as follows:

- `Shapley_Values_for_Measuring_Contribution_in_Influence_fullversion.pdf`: The full version of our research paper, including appendices.
- `data/`: Contains example datasets used in our experiments, such as `congress` and `Twitter`.
- `experiments/`: Contains the Python scripts used to run the experiments described in the paper.
- `output/`: Stores the summarized results from our experiments, including figures and tables.
- `results/`: Contains raw log files and the computed Shapley values for each experiment.
- `shapley_rust_pyo3/`: Contains the core Rust implementation of the Shapley value algorithms.
- `requirements.txt`: A list of the required Python packages for this project.

## Getting Started

### Prerequisites

- Python 3.8+
- Rust (latest stable version recommended)
- A C compiler

### Setup

1.  **Install Python dependencies:**
    The required Python packages are listed in `requirements.txt`. We recommend using a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Compile the Rust module:**
    The core logic is implemented in Rust for performance. Compile it into a Python module using `maturin`:
    ```bash
    cd shapley_rust_pyo3
    maturin develop --release
    cd ..
    ```
    This command builds the Rust code and installs the `shapley_rust` package in your virtual environment, making it available to your Python scripts.
3. **Usage in Python:**
    Once the package is built, you can import and use it in your Python script like any other module:

    ```python
    import shapley_rust
    ```

## Running Experiments

The scripts to reproduce our results are in the `experiments/` directory. They should be run from within that directory.
```bash
cd experiments
```

To run the case study on the Congress dataset:
```bash
python3 exp_casestudy_congress.py
```

To run the experiments for the Twitter dataset:
```bash
python3 exp_twitter.py
```
You can specify which algorithms to run by modifying the `algo_ls` list in the experiment scripts.

## Python API Reference

The `shapley_rust` module provides the following functions:

### Graph Loading
- **`read_undirect_unweight_graph(filename: str) -> PyGraph`**: Reads an undirected and unweighted graph from a file.
- **`read_direct_unweighted_graph(filename: str) -> PyGraph`**: Reads a directed and unweighted graph from a file.
- **`read_direct_weighted_graph(filename: str) -> PyGraph`**: Reads a directed and weighted graph from a file.
- **`read_networkx_edgelist(filename: str) -> PyGraph`**: Reads a graph from a networkx-style edgelist file.

The `PyGraph` object has methods to inspect the graph, such as `node_count()`, `edge_count()`, `get_nodes()`, `get_edges()`, and `add_seeds()`.

### Shapley Value Calculation Algorithms
This section details the functions for Shapley value calculation and maps them to the notation used in our paper.

---
#### `approx_permutation1(graph, parallel, gen_seed=None, max_hop=None, n=None, m=None, epsilon=None, delta=None)`
<!-- ```python
shapley_rust.approx_permutation1(graph, parallel, gen_seed=None, max_hop=None, n=None, m=None, epsilon=None, delta=None, n_jobs=None)
``` -->
- **Paper Notation**: *ApproxPermuteMC*

**Parameters**
- **`graph`**: `PyGraph`  
  The input graph.
- **`parallel`**: `bool`  
  Set to `True` to run the computation in parallel
- **`gen_seed`**: `int`, optional  
  Random seed for sampling generation
- **`max_hop`**: `int`, optional  
  Maximum number of hops for the influence spread simulation. Set to `1` for single-step (one-hop) influence; by default, uses fixed-steps termination.
- **`n`**: `int`, optional  
  Number of permutations samples
- **`m`**: `int`, optional  
  Number of Monte Carlo samples for estimating each influence spread.
- **`epsilon`**: `float`, optional  
  Accuracy parameter.
- **`delta`**: `float`, optional  
  Confidence parameter.

**Returns**
- `dict[int, float]`  
  A dictionary mapping seed node IDs to their Shapley values.

---
#### `approx_permutation2(graph, gen_seed=None, max_hop=None, n=None, epsilon=None, delta=None, parallel=True)`
<!-- ```python
shapley_rust.approx_permutation2(
    graph, 
    gen_seed=None, 
    max_hop=None, 
    n=None, 
    epsilon=None, 
    delta=None, 
    parallel=True
)
``` -->
- **Paper Notation**: *ApproxPermuteDirect*

**Parameters**
- **`graph`**: `PyGraph`  
  <!-- The input graph. -->
- **`gen_seed`**: `int`, optional  
- **`max_hop`**: `int`, optional  
- **`n`**: `int`, optional  
  Number of permutation samples
- **`epsilon`**: `float`, optional  
  Accuracy parameter.
- **`delta`**: `float`, optional  
  Confidence parameter.
- **`parallel`**: `bool`  


**Returns**
- `dict[int, float]`  
  A dictionary mapping seed node IDs to their Shapley values.

---
#### `rr_shapley_approx( graph, k, eps, ell, random_seed=None, num_theta=None, parallel=True, max_hop=None)`
<!-- ```python
shapley_rust.rr_shapley_approx(
    graph, 
    k, 
    eps, 
    ell, 
    random_seed=None, 
    num_theta=None, 
    parallel=True, 
    max_hop=None
)
``` -->
- **Paper Notation**: *ApproxRRSet*

**Parameters**
- **`graph`**: `PyGraph`  
  The input graph.
- **`k`**: `int`  
  Accuracy parameter for the k-th largest Shapley values
- **`eps`**: `float`  
  Accuracy parameter for the approximation
- **`ell`**: `float`  
  Confidence Parameter
- **`random_seed`**: `int`, optional  
  Random seed for sampling generation
- **`num_theta`**: `int`, optional  
  Number of reverse reachable sets to generate.
- **`parallel`**: `bool`  
- **`max_hop`**: `int`, optional  


**Returns**
- `dict[int, float]`  
  A dictionary mapping node IDs to their Shapley values.

---
#### Exact & DP Algorithms
These algorithms take only the graph as input and return a dictionary of Shapley values.

- **`single_step_shapley_rust(graph)`**
  - **Paper Notation**: *ExactSingleStep*
  <!-- - **Description**: An exact algorithm for single-step diffusion models using dynamic programming. -->
- **`single_step_shapley_optimize_rust(graph)`**
  - **Paper Notation**: *ExactSingleStepOpt*
  <!-- - **Description**: An optimized version of the dynamic programming algorithm. -->
- **`exact_single_step_shapley_rust(graph)`**
  - An exact algorithm that computes Shapley values by exhaustively evaluating all possible permutations, following the original Shapley value definition.


