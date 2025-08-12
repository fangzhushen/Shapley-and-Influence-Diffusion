#!/bin/bash
#SBATCH --job-name=shap
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G

# Compile and run with optimizations:
cargo run --release
