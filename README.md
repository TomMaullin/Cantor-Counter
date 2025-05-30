# Cantor-Thirds Counterexample

This repository contains code and supplemental material for the paper *Regularity Conditions for Critical Point Convergence*. The material included here details the construction of a $C^1$ function on the unit ball in $\mathbb{R}^3$ with a single critical point at the origin, whose zero level set intersects every ball $B_\delta$ (for $\delta \in (0,1)$ ) non-transversally. This explicit construction demonstrates the necessity of Lemma 1 in the main text, and we refer to it as the *Cantor-thirds counterexample*.

## 📁 Contents

- [`cantor_counter.ipynb`](cantor_counter.ipynb)  
  A Jupyter notebook that walks through the construction, provides visualizations of the key functions ($H^\ast$, $f^\ast$, and $f$), and contains supplementary proofs of all relevant properties.

- [`src/`](src/)  
  Source code used to define and visualize the core functions in the construction:
  
  - `H_construct.py` – Provides functions to evaluate the $H^\ast$, $\tilde{H}$ and $h$.
  - `T_construct.py` – Constructs the transition function $T^\ast$.
  - `coord_transform.py` – Contains a helper function for converting spherical coordinates to Cartesian coordinates.
  - `extend.py` – Provides a function which extends $C^1$ functions defined on $[-1,1]^2$ to be constant outside $[-2,2]^2$.
  - `f_construct.py` – Assembles the final function $f^\ast$ and wraps it into $f$ in spherical coordinates.
  - `integral.py` – Supporting function for numerical integration.
  - `plot.py` – Contains all utilities for generating the figures used in the paper and supplemental material.

## 📜 Mathematical Summary

The constructed function $f : B_1 \subset \mathbb{R}^3 \rightarrow \mathbb{R}$ possesses the following properties:
- $f$ is $C^1$
- $f$ has a single critical point at the origin
- $f^{-1}(0)$ tangentially intersects every ball $B_\delta$ for $\delta \in [0,1]$.

The construction uses properties of the Cantor ternary set to densely place critical points across all levels.

## 📊 Figures

All figures in the Section 6 of the supplement to the main text are generated directly from the code in this repository, including:
- The construction of $H^*$ from $\tilde{H}$,
- 2D-to-polar and Cartesian-to-spherical transformations,
- Slices of $f^*$ and $f$ visualized over $\mathbb{R}^3$.

## 🔧 Requirements

The notebook uses the following Python packages:
- `numpy`
- `matplotlib`
- `plotly`
- `scipy`
- `notebook` or `jupyterlab` (for running locally)
