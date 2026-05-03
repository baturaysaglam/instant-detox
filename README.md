# Test-Time Detoxification without Training or Learning Anything

<p align="center">
  <img src="figs/tide.jpg" width="75%">
</p>

This repository contains the code for the paper *"Test-Time Detoxification without Training or Learning Anything."*

We introduce **Test-Time Iterative Detoxification via Embeddings (TIDE)**, a method for reducing toxic outputs from language models at inference time. Unlike existing approaches, TIDE requires no retraining, fine-tuning, or auxiliary learned components—it operates entirely at test time by performing zeroth-order gradient descent directly on prompt embeddings to minimize output toxicity.

### Algorithm in Action — **Warning:** Inappropriate and offensive language

<p align="center">
  <img src="figs/example_optimization.png" width="60%">
</p>

## Repository Structure

- **`cookbook.ipynb`** — An interactive notebook demonstrating how TIDE works step-by-step. Use this to explore the method and experiment with different prompts and hyperparameters.

- **`generate_base.py`** — Collects baseline completions and toxicity scores from an unmodified model. Run this first to establish baseline performance.

- **`generate_tide.py`** — Runs TIDE's optimization procedure on prompts to generate detoxified completions.

- **`compute_metrics.py`** — Computes evaluation metrics (toxicity scores, perplexity) for model responses. Built on vLLM and easily adaptable for other detoxification experiments.

- **`responses/`** — Stored model responses on the challenging subset of the RealToxicityPrompts (RTP) benchmark, provided as a reference.

- **`utils/`** — Core utilities including model loading, TIDE optimization functions, and toxicity scoring via the Perspective API.

## Usage

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Collecting Baseline Responses

Generate baseline completions from an unmodified model:

```bash
python generate_base.py --model google/gemma-2-2b --dataset rtp
```

This saves responses to `responses/baselines/`. Run `python generate_base.py --help` for all available arguments.

## Running TIDE

Run TIDE optimization on toxic prompts:

```bash
python generate_tide.py --model google/gemma-2-2b --dataset rtp
```

Results are saved to `responses/tide/`.

### TIDE Optimization Parameters

TIDE performs zeroth-order gradient descent directly on prompt embeddings to minimize output toxicity. The following parameters control this optimization procedure:

#### a. Prompt Filtering

| Argument | Description | Default |
|----------|-------------|---------|
| <nobr>`--toxic-th`</nobr> | Toxicity threshold for selecting which prompts to optimize. Only prompts whose baseline completions meet or exceed this toxicity score are processed by TIDE. Set to `-1` to optimize all prompts regardless of baseline toxicity. Higher values (e.g., 0.9) focus compute on the most challenging cases. | `0.9` |
| <nobr>`--toxic-th-mode`</nobr> | Determines how baseline toxicity is computed when filtering. `"mean"` uses the average toxicity across all K baseline completions; `"max"` uses the highest toxicity score. Use `"max"` for stricter filtering that catches any toxic generation. | `mean` |

#### b. Zeroth-Order Optimization

| Argument | Description | Default |
|----------|-------------|---------|
| <nobr>`--num-iter`</nobr> | Maximum number of optimization iterations per prompt. Each iteration computes a gradient estimate and updates the embedding. More iterations allow stronger detoxification but increase compute cost linearly. | `10` |
| <nobr>`--mu`</nobr> | Perturbation scale (smoothing parameter) for gradient estimation. Controls the radius of random perturbations used to approximate the gradient via finite differences. Smaller values give more accurate gradients but may be noisier; larger values smooth over local minima but reduce precision. | `0.1` |
| <nobr>`--N`</nobr> | Number of random perturbation directions sampled per gradient estimate. More samples reduce variance in the gradient approximation at the cost of additional forward passes. The total forward passes per iteration equals 2×N (positive and negative perturbations). | `16` |
| <nobr>`--stepsize`</nobr> | Learning rate for the embedding update. After computing the normalized gradient, the embedding moves by `stepsize × gradient` in the descent direction. Larger values enable faster convergence but risk overshooting; smaller values are more stable but slower. | `1.0` |

#### c. Semantic Preservation

| Argument | Description | Default |
|----------|-------------|---------|
| <nobr>`--cosine-sim-th`</nobr> | Minimum cosine similarity between optimized and original embeddings. After each gradient step, if the cosine similarity drops below this threshold, the embedding is projected back onto the constraint boundary. This prevents semantic drift—ensuring the optimized prompt remains meaningfully related to the original. Set to `0.0` to disable projection (no constraint). Higher values (e.g., 0.9) enforce tighter semantic fidelity. | `0.0` |

#### d. Early Stopping    

| Argument | Description | Default |
|----------|-------------|---------|
| <nobr>`--early-stopping-th`</nobr> | Toxicity threshold for early termination. If the output toxicity drops below this value during optimization, the procedure stops early for that prompt. This saves compute on prompts that are successfully detoxified before reaching `--num-iter` iterations. | `0.5` |


### An Example TIDE Configuration

The following mirrors our experimental setup for Llama 3.1-8B:

```bash
python generate_tide.py \
    --model meta-llama/Llama-3.1-8B \
    --dataset rtp \
    --num-iter 10 \
    --N 16 \
    --mu 0.03 \
    --stepsize 0.3 \
    --cosine-sim-th 0.2 \
    --early-stopping-th 0.5 \
    --toxic-th -1
```

This configuration uses a smaller perturbation scale and step size suited for larger models, applies a mild semantic constraint, and optimizes all prompts regardless of baseline toxicity.

## Computing Metrics

Evaluate toxicity and fluency metrics on generated responses:

```bash
python compute_metrics.py --results-path responses/tide/gemma-2-2b/rtp.json
```

## Interactive Exploration

For a hands-on walkthrough of the method, open the cookbook notebook:

```bash
jupyter notebook cookbook.ipynb
```

The notebook demonstrates the full optimization loop on individual prompts, making it easy to understand how TIDE reduces toxicity iteratively.
