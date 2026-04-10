# Step-MPPI

A learning-to-optimize (L2O) framework for Model Predictive Path Integral (MPPI) control, featuring Step-MPPI (DMPPI). This repository implements the proposed Step-MPPI to autonomous racing (F1Tenth) and quadrupedal locomotion tasks.

**Paper:** [arXiv:2604.01539](https://arxiv.org/abs/2604.01539)

## Overview

Step-MPPI is a framework that uses machine learning to optimize the sampling distribution in MPPI controllers, leading to improved performance and efficiency. The repository includes implementation for the following methods:

- **Step-MPPI (DMPPI)**: Learns optimal sampling distributions for MPPI
- **DPC**: Differentiable Predictive Control with learned policies
- **Neural Step-MPPI (DMPPI)**: Learns optimal sampling distributions and update rules for MPPI
- **Applications**: [F1Tenth](f1tenth/) autonomous racing and [Quadruped](quadruped/) robot control

## Quick Start

For detailed installation instructions and how to run the examples, please refer to:

- [F1Tenth Setup](f1tenth/README.md)
- [Quadruped Setup](quadruped/README.md)

## Citation

If you use this code in your research, please cite:

```
@article{le2026toward,
  title={Toward Single-Step MPPI via Differentiable Predictive Control},
  author={Le, Viet-Anh and Tumu, Renukanandan and Mangharam, Rahul},
  journal={arXiv preprint arXiv:2604.01539},
  year={2026}
}
```

## Contact

For questions or collaborations, please contact the authors:

- **Viet-Anh Le**: [vietanh@seas.upenn.edu](mailto:vietanh@seas.upenn.edu)

## Acknowledgments

- Uses forked versions of [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym) and [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)
