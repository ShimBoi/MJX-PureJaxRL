# MJX-PureJaxRL

[](https://github.com/ShimBoi/MJX-PureJaxRL#mjx-purejaxrl)

Adapting the PureJaxRL framework to use MJX manipulation environments with Madrona GPU rendering. I chose to undertake this small project to quickly train and experiment with robotic manipulation tasks with jax.

## How to run the training script

[](https://github.com/ShimBoi/MJX-PureJaxRL#how-to-run-the-training-script)

```
cd purejaxrl/purejaxrl
python mujoco_ppo.py --job train

for eval:
python mujoco_ppo.py --job eval
```

## Checkpoints/Videos

[](https://github.com/ShimBoi/MJX-PureJaxRL#checkpointsvideos)

- checlpoints are in purejaxrl/purejaxrl/checkpoints
- rollout videos are in purejaxrl/purejaxrl/videos

## Timing

[](https://github.com/ShimBoi/MJX-PureJaxRL#timing)

- ~15min on an RTX4090 for ~50mil env steps
- rendering avg total time 0.258411ms
