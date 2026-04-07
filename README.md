# Luoshu Kit V0.2 CNN -> ResNet

## LuoshuKit is a plug-in. Inject it into any model.

## A mechanistic interpretability layer — every value now has a computed address instead of being searched.

LuoshuKit implements a structured addressing layer for neural representations,
where internal values are assigned addresses that can be directly decoded rather than located through search.

---

Code for the paper: *From Search to Computation: Structured Error Tracing via Recursive Luoshu Localization*.

## Main files

* `run_experiment.py`: main script for A0/A1/A2 comparison and trace-form validation
* `model_setup.py`: model and structured localization setup
* `tracing_cost_analysis.py`: tracing cost and trace analysis

## What it shows

The code compares three tracing regimes:

* A0: full search
* A1: anchor-only guided search
* A2: anchor–path structured computation

Typical behavior:

* A0 tracing cost: 36
* A1 tracing cost: 18
* A2 follows a fixed anchor-plus-decode trace

## Run

```bash
python run_experiment.py
```
