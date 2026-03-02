# Neural Likelihood Ratio Estimation

## Abstract

The determination of transverse single-spin asymmetries in experiments involving polarized targets and/or beams may encounter challenges when (1) the magnitude of the polarization varies greatly with time, (2) the polarization magnitude is not the same for each spin state, (3) different integrated luminosities occur for different spin states or different target materials, and/or (4) some kinematic variables require unfolding; these are just a few examples. We present general methods of determining the asymmetry based on both binned analysis and unbinned maximum likelihood optimization, incorporating the unfolding of kinematic variables that are smeared by detector effects, and also including the possibility of background subtraction.

This work is published in [arXiv:2602.02325](https://arxiv.org/abs/2602.02325).

## Learning the Likelihood Ratio via Classification

Estimating the likelihood ratio is a challenging task in real-world scenarios, particularly when the underlying densities are unknown. However, this can be achieved using a binary classifier.

Consider a binary classifier $s(x)$ trained to distinguish between a **source distribution** (assigned label $y=0$) and a **target distribution** (assigned label $y=1$). The estimated likelihood ratio $\hat{r}(x)$ can be expressed as:

$$\hat{r}(x) = \frac{s(x)}{1 - s(x)}$$


## Analysis Environment

This repository has been tested on the **Elastic Analysis Facility** ([EAF](https://arxiv.org/abs/2506.08222)) at Fermilab, using `AL9` (AlmaLinux 9) with the `LCG_105_cuda` software stack.

To reproduce the analysis, execute the following shell script:

```bash
./tutorial/run_tutorial.sh
````

## Roadmap

  - [ ] Implement the OmniFold algorithm.

## References

1.  **A Guide to Constraining Effective Field Theories with Machine Learning** Brehmer et al. [arXiv:1805.00020](https://arxiv.org/abs/1805.00020)

2.  **Approximating Likelihood Ratios with Calibrated Discriminative Classifiers** Cranmer et al. [arXiv:1506.02169](https://arxiv.org/abs/1506.02169)

3.  **A Practical Guide to Unbinned Unfolding** [arXiv:2507.09582](https://arxiv.org/abs/2507.09582)

4.  **Machine Learning-Assisted Unfolding for Neutrino Cross-section Measurements with the OmniFold Technique** [arXiv:2504.06857](https://arxiv.org/abs/2504.06857)

5.  **OmniFold: A Method to Simultaneously Unfold All Observables** Andreassen et al. [arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
