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
@misc{step-mppi-2024,
  title={Step-MPPI: A Learning-to-Optimize Framework for Model Predictive Path Integral Control},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/Step-MPPI}
}
```

## Contact

For questions or collaborations, please contact the authors:

- **Viet-Anh Le**: [vietanh@seas.upenn.edu](mailto:vietanh@seas.upenn.edu)

## Acknowledgments

- Uses forked versions of [F1Tenth Gym](https://github.com/vietanhle0101/f1tenth_gym) and [Quadruped-PyMPC](https://github.com/vietanhle0101/Quadruped-PyMPC)
