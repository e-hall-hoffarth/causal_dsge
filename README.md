# Causal Discovery of Macroeconomic State-Space Models

This repository contains all of the code and data used to implement the study as well as the paper itself.

### Abstract

This paper presents a set of tests and an algorithm for agnostic, data-driven selection among macroeconomic DSGE models inspired by structure learning methods for DAGs. Structure learning algorithms can be used because the log-linear state-space solution to any DSGE model is also a DAG. In particular, it is possible to define a small set of conditional independence relationships which uniquely identify the ground-truth model that is consistent with some underlying DGP. The algorithm tests feasible analogues of these criteria against the set of possible state-space models in order to consistently estimate the ground-truth model. In small samples where the result may not be unique, conditional independence tests can be combined with likelihood maximisation in order to select a single optimal model. The efficacy of this algorithm is demonstrated for simulated data, and results for real data are also provided and discussed.
