# Pose-Selective Sparse G-CNNs

Pose-Selective Sparse G-CNNs implement differentiable group subset discovery for equivariant convolutional networks. The models learn to gate orientation channels so that only a subset of group elements is active, reducing computation while retaining the benefits of equivariance.

## Repository layout

- **src/** – training code, model definitions and utilities
- **configs/** – YAML files describing datasets, model hyperparameters and schedules
- **baselines/** – dense baseline models and a script for FLOP calculation
- **scripts/** – helper scripts for running or submitting jobs
- **results/** – example training logs and saved checkpoints

## Installation

Create a Python environment with PyTorch and other dependencies:

```bash
pip install torch torchvision e2cnn PyYAML tqdm thop
```

Experiments download datasets automatically to the path given in each config (default `./data`).

## Training

Pick a configuration and random seed and run

```bash
python src/train.py --config configs/cifar10.yaml --seed 0
```

The script logs training progress and saves the best model and metrics under `results/<DATASET>/<MODEL>/seed_<SEED>`.

## Baseline FLOPs

To compute FLOPs for dense baselines run

```bash
python baselines/calculate_baseline_flops.py --config configs/cifar10.yaml
```

## HPC scripts

`scripts/run_experiment.sh` wraps the training command for a Slurm cluster and `scripts/submit_all.sh` submits jobs for all datasets and seeds while respecting queue limits.

## License

This project is licensed under the Apache 2.0 License; see the [LICENSE](LICENSE) file for details.

## Contact / Support

For any questions, issues, or collaboration inquiries related to this project, please feel free to contact the authors:

- **Dr. S. Pradeep** – Post-Doctoral
  Researcher and Principal Investigator at the Machine Intelligence Lab, Department of Computer
  Science and Engineering, IIT Roorkee

  Email: pradeep.cs@sric.iitr.ac.in

- **Kanishk Sharma** – M.Tech, Department of Computer Science and Engineering, IIT Roorkee  
  Email: kanishk_s@cs.iitr.ac.in

- **Dr. R. Balasubramanian** – Professor (HAG) and Head of the Department of Computer Science & Engineering, IIT Roorkee  
  Email: bala@cs.iitr.ac.in

We welcome feedback, bug reports, and contributions!
