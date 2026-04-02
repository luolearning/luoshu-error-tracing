# Luoshu Error Tracing

Code for the paper: *From Search to Computation: Structured Error Tracing via Recursive Luoshu Localization*.

## Main file

- `main.py`: main script for trace-form validation and A0/A1/A2 comparison
- `clean_v6.py`: cleaned implementation utilities
- `step4_search_cost_v6_trace.py`: search-cost and trace analysis

## What it shows

The code compares three tracing regimes:

- A0: full search
- A1: anchor-only guided search
- A2: anchor-path structured computation

Typical behavior:

- A0 trace length: 36
- A1 trace length: 18
- A2 follows a fixed anchor-plus-decode trace

## Run

```bash
python main.py
