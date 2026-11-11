# Drone Simulator: QuadSwarm

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

[//]: # (**Paper:** https://arxiv.org/abs/2109.07735)

[//]: # ()
[//]: # (**Website:** https://sites.google.com/view/swarm-rl)


<p align="middle">

<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Swarm_vs_Swarm.gif?raw=true" width="45%">
</p> 

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Obstacles_Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Scale_32_Static_Same_Goal.gif?raw=true" width="45%">
</p> 

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions >=3.11 are supported):

```
git clone https://github.com/eliotshekhtman/quad-swarm-rl.git
cd quad-swarm-rl

conda create -n swarm-cbf python=3.11 "numpy<2" "scipy<1.14" "cvxpy==1.4.2" osqp scs ecos -c conda-forge
conda activate swarm-cbf
pip install -e .
conda install -c conda-forge ffmpeg
pip install pybind11 --no-deps
conda install -c conda-forge libiconv
```

This should pull and install all the necessary dependencies including PyTorch.

## Running experiments

### Conformal CBF evaluation

`conformal.py` is the single driver for the supported radius-update
procedures. Select one with `--update-procedure`:

```
python conformal.py [model and output arguments] --update-procedure robust
python conformal.py [model and output arguments] --update-procedure naive
python conformal.py [model and output arguments] --update-procedure non-interactive
python conformal.py [model and output arguments] --update-procedure calibrate-once
python conformal.py [model and output arguments] --update-procedure calibrate-once-paper
python conformal.py [model and output arguments] --update-procedure anytime --anytime-init-trajectories 1000
python conformal.py [model and output arguments] --update-procedure anytime-nonparametric --anytime-init-trajectories 1000
```

`robust` implements the adaptive robust update, `naive` sets the next radius
directly from the current conformal quantile, `non-interactive` reproduces the
final historical `conformal_falldown.py` schedule (calibrate at the known-safe
initial radius, update once to that quantile, and then stop), and
`calibrate-once` performs the initial calibration followed by the fixed legacy
recurrence. `calibrate-once-paper` instead solves `r = q0 + kappa * abs(r - r0)`
once, as specified by One-time CP in the paper, and keeps that radius fixed.
Its second collected batch measures the resulting quantile but does not update
the radius; `--num_episodes` does not affect this method. The metrics can be
passed to `plot_combined.py --calibrate_once` like the legacy baseline.
All modes use the fourth-order exponential control barrier function
and execute rollouts sequentially. Run `python conformal.py --help` for the
required model and output arguments.

`--seed` controls the scenario startup and prediction-artifact validation.
Optionally add `--run-seed 100` to seed calibration and assessment randomness
once, after scenario and prediction setup, for an independent repetition of
the same scenario. Keep the same `--seed` and `--init_predictions` across
repetitions. No environment reset is performed at this boundary, and the
random stream continues across trajectories and episodes.

Omitting `--run-seed` preserves the historical random stream exactly: there
is no additional seeding call. Explicitly setting it equal to `--seed` does
**not** reproduce the omitted-option behavior, since startup has already
consumed random draws. Both values are saved in `conformal_args.json`.
Use a distinct `--experiment_name` for each repetition.

`run_all.sh` also accepts `--run-seed`. When supplied, it appends `_run<seed>`
to all experiment names and log filenames, while reusing the prediction
artifact for the scenario seed. Without this option, names and commands
are unchanged. Equal run seeds across different algorithms do not ensure
paired assessments, because their sample counts and RNG consumption differ.

Policy inference defaults to CPU. Use `--device gpu` together with
`CUDA_VISIBLE_DEVICES=<index>` to assign a process to one GPU; simulation and
ECBF optimization remain CPU-bound.

Controller settings are independent of the radius-update `--kappa`:
`--cbf-k1` defaults to 2.5 and `--cbf-solver` defaults to `cascade`.
For historical-controller replay, use `--cbf-k1 4 --cbf-solver legacy`.
This restores ECOS-only solving, accepts inaccurate optimal solutions, and
falls back to an available iterate or nominal thrust rather than previous thrust.
It is a reproduction mode, not a stronger safety guarantee. Both controller
settings are recorded in the experiment arguments and metrics.

`anytime` implements the revised Algorithm 1 with the likelihood-based
Jeffreys-mixture e-process. `anytime-nonparametric` uses the normal-mixture
e-process with the same quantile confidence sequence, bracket update, and
stopping rule. One score represents the complete joint rollout: it is the
physical separation radius plus the largest prediction error over all agents
and timesteps. `--num_trajectories` is the per-episode cap in these modes.

Select the other paper mixtures with `--update-procedure anytime` and
`--anytime-family informed`, `jeffreys-informed`, or `directional`.
`--anytime-family normal` is also an alias for the nonparametric family;
omitting the family preserves Jeffreys. Hybrid and directional mixtures use
`--anytime-jeffreys-weight` (eta, default 0.5). Directional additionally uses
`--anytime-directional-weight` (w, default 0.5), the weight on the normalized
lower-side mixture. The entire hybrid prior is split at alpha before weighting
the two sides, as in Appendix A.2.

These three new families require `--anytime-init-trajectories N`: equation (24)
initializes the first Beta prior from the independent initial calibration and
each subsequent prior from only the preceding episode's actual calibration
scores. The exceedances are evaluated at the new current radius, with shift
`kappa * abs(previous_radius - current_radius)` (zero initially). The prior is
frozen for the entire episode, shared by all threshold tests, and previous
scores are not reused as new evidence. No extra collection is needed for this
initialization. The metrics include `anytime_family` and per-episode
`anytime_prior_*_per_episode` arrays (a, b, mean, concentration, source count and
radius, shift, nearby-score fraction, and weights) for reconstruction.

All methods save `assessment_radii`, `assessment_scores`, and
`assessment_quantiles`. Each score is physical separation plus the maximum
prediction error over all non-ego agents and rollout timesteps. Quantiles use
the empirical `1-alpha` order statistic (`inverted_cdf`), not the adjusted
calibration level. The old `qj_per_episode` is unchanged. Assessment arrays
include both initial and final policies: robust, naive, and anytime prepend
the initial policy to their post-update assessments; non-interactive and both
calibrate-once variants already assess the initial policy. Repeated radii are
retained. The arrays explicitly identify the radius for every assessment row.

All anytime families also save `anytime_reference_scores` and
`anytime_reference_quantile_per_episode` for the within-episode bounds plot.
Rows correspond to `anytime_current_radius_per_episode`, not the updated radius.
These are independent assessment scores and their empirical `1-alpha` quantiles,
not adjusted conformal quantiles or true population quantiles. Existing post-update
assessments supply subsequent episodes' references. Robust, naive, and anytime
collect one extra `num_eval_trajs` batch at the initial radius after all updates,
using the original predictor even when baseline prediction updates are enabled.
This leaves the algorithm's
random sequence and decisions unchanged. These assessment samples are not
calibration samples. Existing metrics files are not backfilled.
The bounds plots from `plot_calibration.py` and `plot_conformal.py` automatically
show the saved independent quantile estimate as a black dotted line and the
deployed radius as a grey dash-dot line. Older files omit the quantile reference.

The anytime driver uses fixed quantile sensitivity `--kappa` (default 0.6); density
bounds and probability-slope estimates are not part of the revised algorithm.
An anytime run must either certify its initial radius with independent data via
`--anytime-init-trajectories N` and `--anytime-delta-init`, or explicitly state
the known-safe assumption with `--anytime-assume-initial-safe`. Prediction
updates are rejected because they change the fixed radius-indexed policy family
required by the guarantee. The default stopping-check frequency is one complete
joint rollout and can be batched with `--anytime-check-every`.

### Compact Paper Figures

`plot_compact.py` accepts the same six input flags as `plot_combined.py`:
`--robust`, `--naive`, `--calibrate_once`, `--non_interactive`, `--anytime`, and
`--anytime_nonparametric`, each followed by a metrics `.npz` path. Use
`--output_dir` to select the destination. It leaves `plot_combined.py` unchanged.
Optional inputs `--anytime_informed`, `--anytime_jeffreys_informed`, and
`--anytime_directional` add those saved mixtures. The plotter checks each file's
family metadata, keeps the original six-input invocation valid, and expands the
trajectory comparison to accommodate the supplied methods.

The 2-by-4 `episode_progression.pdf` separates CP baselines from the supplied
anytime mixtures. `--second-panel qj` selects radius, independent quantile,
joint tube coverage, and cumulative calibration trajectories. The default
`--second-panel reward` selects reward for column two and overlays independent
quantiles as same-color dotted lines on column one's solid radii, labeled
`r_j, q_j`. The quantile-column mode requires the new assessment fields in all
input files; reward mode warns and omits unavailable dotted lines in old files.
It never substitutes adjusted calibration quantiles. Error bars retain the existing per-rollout reward
quantiles, not confidence intervals. Solid sample-budget lines count update
calibration; dashed lines also include initial safety calibration. Assessment
and prediction-collection rollouts are not included. Episode indexing and
terminal-value repetition match the combined plotter; percentages are scaled
to 0-100. Legacy calibrate-once is labeled separately from paper One-time CP.

Four separate supplements are written: `ego_safety.pdf`,
`anytime_quantile_bounds.pdf`, `final_trajectories.pdf`, and
`anytime_tube_progression.pdf`. The bounds figure uses ordinary anytime only;
`--bounds_episodes 0 2 4` overrides first/middle/last selection. Missing independent
references are omitted with a warning, never replaced by calibration quantiles
or radii. Infinite bounds are not replaced by finite values. The tube figure
uses the first, second, and final saved post-update rollouts, not an invented
initial-policy rollout. `--tube_agent` selects a non-ego agent (default 0), and
`--elev` / `--azim` set a common 3D viewpoint. The ego is the last stored agent.
Missing supplementary data skips the affected figure with a warning.
Computer Modern fonts, paper colors/markers, and boxed axes are scoped to this
plotter. LaTeX is used when installed; otherwise bundled Computer Modern fonts
and mathtext render the figures without a TeX installation.

### Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

We provide a training script `train.sh`, so you can simply start training by command `bash train.sh`.

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. These runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

This command will start training four different seeds in parallel on a 4-GPU server. Adjust the parameters accordingly to match
your hardware setup.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```
### WandB support

If you want to monitor training with WandB, follow the steps below: 
- setup WandB locally by running `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb).
* add `--with_wandb=True` in the command.

Here is a total list of wandb settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

### Test
To test the trained model, run the following command:

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

CAMERA_VIEWS can be any number of views from the following: `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`


## Unit Tests

To run unit tests:

```
./run_tests.sh
```

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our following papers.

### QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control 
[ICRA Workshop: The Role of Robotics Simulators for Unmanned Aerial Vehicles, 2023](https://imrclab.github.io/workshop-uav-sims-icra2023/)

Drone Simulator for Reinforcement Learning.
```
@article{huang2023quadswarm,
  title={Quadswarm: A modular multi-quadrotor simulator for deep reinforcement learning with direct thrust control},
  author={Huang, Zhehui and Batra, Sumeet and Chen, Tao and Krupani, Rahul and Kumar, Tushar and Molchanov, Artem and Petrenko, Aleksei and Preiss, James A and Yang, Zhaojing and Sukhatme, Gaurav S},
  journal={arXiv preprint arXiv:2306.09537},
  year={2023}
}
```

### Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors
IROS 2019

Single drone: a unified control policy adaptable to various types of physical quadrotors.
```
@inproceedings{molchanov2019sim,
  title={Sim-to-(multi)-real: Transfer of low-level robust control policies to multiple quadrotors},
  author={Molchanov, Artem and Chen, Tao and H{\"o}nig, Wolfgang and Preiss, James A and Ayanian, Nora and Sukhatme, Gaurav S},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={59--66},
  year={2019},
  organization={IEEE}
}
```
### Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning
CoRL 2021

Multiple drones: a decentralized control policy for multiple drones in obstacle free environments.
```
@inproceedings{batra21corl,
  author    = {Sumeet Batra and
               Zhehui Huang and
               Aleksei Petrenko and
               Tushar Kumar and
               Artem Molchanov and
               Gaurav S. Sukhatme},
  title     = {Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning},
  booktitle = {5th Conference on Robot Learning, CoRL 2021, 8-11 November 2021, London, England, {UK}},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.07735}
}
```
### Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning
ICRA 2024

Multiple drones: a decentralized control policy for multiple drones in obstacle dense environments.
```
@inproceedings{huang2024collision,
  title={Collision avoidance and navigation for a quadrotor swarm using end-to-end deep reinforcement learning},
  author={Huang, Zhehui and Yang, Zhaojing and Krupani, Rahul and {\c{S}}enba{\c{s}}lar, Bask{\i}n and Batra, Sumeet and Sukhatme, Gaurav S},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={300--306},
  year={2024},
  organization={IEEE}
}
```
### HyperPPO: A scalable method for finding small policies for robotic control
ICRA 2024

A method to find the smallest control policy for deployment: train once, get tons of models with different size by using HyperNetworks.

We only need four neurons to control a quadrotor! That is super amazing!

Please check following videos for more details:
- [Square Grid Trajectory](https://www.youtube.com/watch?v=IenGT_TOwGQ&ab_channel=USCRESL)
- [Bezier Curve Trajectory](https://www.youtube.com/watch?v=B5EpKlD5F68&ab_channel=USCRESL)
```
@inproceedings{hegde2024hyperppo,
  title={Hyperppo: A scalable method for finding small policies for robotic control},
  author={Hegde, Shashank and Huang, Zhehui and Sukhatme, Gaurav S},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10821--10828},
  year={2024},
  organization={IEEE}
}
```

### Tiny End-to-End Navigation for Quadrotor Swarms
ICRA 2025 Workshop

An end-to-end system for navigating a quadrotor swarm with severely resource-constrained hardware, fully relying on onboard resources for perception, localization, planning, and control.

```
@article{chiutiny,
  title={Tiny End-to-End Navigation for Quadrotor Swarms},
  author={Chiu, Darren and Huang, Zhehui and Ge, Ruohai and Sukhatme, Gaurav S}
}
```

### Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation
Under Review

A novel framwork for inference-time behavior steering of pre-trained robot policies without retraining or fine-tuning.
```
@article{das2025latent,
  title={Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation},
  author={Das, Satyajeet and Chiu, Darren and Huang, Zhehui and Lindemann, Lars and Sukhatme, Gaurav S},
  journal={arXiv preprint arXiv:2509.20623},
  year={2025}
}
```


### The next paper will appear soon. Hopefully, it will be the end of 2025. That would be a big surprise. 
### After that paper publish, we will spend more time on the community, including maintain a website to cover more details of using this simulator and open source some important trained models. 

Github issues and pull requests are welcome.
