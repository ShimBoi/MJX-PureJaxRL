# MJX-PureJaxRL
Adapting the PureJaxRL framework to use MJX manipulation environments with Madrona GPU rendering. I chose to udnertake this small project to quickly train and experiment with robotic manipulation tasks with jax.

## How to run the training script
```
cd purejaxrl/purejaxrl
python mujoco_ppo.py --job train

for eval:
python mujoco_ppo.py --job eval
```

## Checkpoints/Videos
- checlpoints are in purejaxrl/purejaxrl/checkpoints
- rollout videos are in purejaxrl/purejaxrl/videos

## Timing
- ~15min on an RTX4090 for ~50mil env steps
- rendering avg total time 0.258411ms

[![Demo Video](https://img.shields.io/badge/▶️-Watch%20Demo-blue)](purejaxrl/purejaxrl/videos/rollout_episode_3_return_15.29.mp4?raw=true)
