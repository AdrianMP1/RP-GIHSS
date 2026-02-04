# Reference-Point-Specification-GIHSS

This repository provides the implementation of a **greedy inclusion algorithm** for **greedy inclusion hypervolume-based subset selection**, together with utilities for generating and visualizing **Lamé (superellipse) curves** at high resolution.

The code accompanies a published research work and was used to produce results and figures reported in the paper.

> **[Reference Point Specification in Greedy Inclusion Hypervolume-based Subset Selection: A Study on Two Objectives]**
> Authors,
> [Full Paper Link.](https://dl.acm.org/doi/10.1145/3712256.3726438)

---

## Structure
- `greedy_algorithm.py`
  Implements a greedy inclusion scheme to solve a subset selection problem.
  The algorithm incrementally builds a solution by adding elements whose hypervolume contribution is the largest at that moment.

- `curve_generation.py`
  Generates Lamé curves with configurable resolution (e.g., `n=10,000`).

---
## Usage
<p align="center">
    <img src="/assets/LAME.gif" width="300" alt="Lamé curves.">
</p>