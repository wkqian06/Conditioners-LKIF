# Conditioners-LKIF
Benchmarking Conditioners in Liang–Kleeman Information Flow: Application to Land–Atmosphere Interactions

This repository provides the MATLAB and Python code used in the analysis presented in:

Benchmarking Conditioners in Liang–Kleeman Information Flow: Application to Land–Atmosphere Interactions
Siddique et al., 2026 (submitted to Earth System Dynamics)

Repository Structure

Multi_LKIF: MATLAB functions and scripts for computing multivariate LKIF, including time-varying estimation with Kalman filter. All functions used for calculating information flow (IF), are included here.

ANOVA.ipynb: Python notebook for running the regime-based ANOVA of ΔIF, quantifying drivers of divergence between bivariate and multivariate causal estimates.

Toy Model.ipynb: Demonstrates theoretical synthetic VAR model experiments under hidden confounding (Appendix A of paper).

rIF_deltaIF.ipynb: Script for calculating |rIF| and |ΔIF|, visualized via split-triangle heatmaps.

Conditioner_Based_Couplings_Analysis.ipynb: Computes the Mediator Dominance Index (MDI), Moderation Gain (MG), Confounding Pressure (CP), and Convergence Rate (CR).

