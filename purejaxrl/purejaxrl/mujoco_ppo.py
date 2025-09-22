import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update('jax_enable_x64', False)
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from mujoco_wrapper import MJXWrapper
from wrappers import (
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from mujoco_wrapper import LogWrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime
import orbax.checkpoint as ocp
import pickle

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "tanh":
            activation = nn.tanh
        elif self.activation == "relu":
            activation = nn.relu
        else:
            raise ValueError("Invalid activation")
        # -------------------------
        # CNN feature extractor
        # -------------------------
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        
        x = x.reshape((x.shape[0], -1))  # flatten
        # -------------------------
        # Actor head
        # -------------------------
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        actor_logstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        # -------------------------
        # Critic head
        # -------------------------
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = MJXWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    # env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def save_checkpoint(train_state, step, checkpoint_dir="checkpoints"):
        """Save training checkpoint with all necessary info"""
        params = train_state.params
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}_{timestamp}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        def jax_to_numpy(pytree):
            return jax.tree_util.tree_map(lambda x: np.array(x) if hasattr(x, 'shape') else x, pytree)
        
        params = jax_to_numpy(params)
        with open(os.path.join(checkpoint_path, "params.pkl"), 'wb') as f:
            pickle.dump(params, f)

def load_checkpoint(checkpoint_path):
        """Load the training state from checkpoint"""
        with open(os.path.join(checkpoint_path, "params.pkl"), 'rb') as f:
            params = pickle.load(f)
        
        def numpy_to_jax(pytree):
            return jax.tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, pytree)
        
        return numpy_to_jax(params)

##########
## EVAL ##
##########

def create_video_rollout(params, config, num_episodes=3, max_steps=200):
    """Create video rollouts after training (non-JIT compiled)"""
    
    # Create environment without vectorization for rendering
    env, env_params = MJXWrapper(config["ENV_NAME"], num_envs=1, train=False), None
    env = LogWrapper(env)
    env = ClipAction(env)
    # Note: Don't use VecEnv or normalization wrappers for rendering
    
    network = ActorCritic(
        env.action_space(env_params).shape[0], 
        activation=config["ACTIVATION"]
    )
    
    videos = []
    episode_returns = []
    
    print(f"Creating {num_episodes} video rollouts...")
    
    for episode in range(num_episodes):
        frames = []
        episode_return = 0.0
        rng = jax.random.PRNGKey(episode + 42)  # Different seed per episode
        
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, env_params)
        
        for step in range(max_steps):
            frames.append(np.array(obs).squeeze())
            pi, value = network.apply(params, obs)
            action = pi.mode()
            action = jnp.ravel(action) 
            
            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            
            episode_return += float(reward)
            
            if done:
                break
        
        if frames:
            videos.append(frames)
            episode_returns.append(episode_return)
            print(f"Episode {episode + 1}: {len(frames)} frames, return: {episode_return:.2f}")
        else:
            print(f"Episode {episode + 1}: No frames captured")
    
    return videos, episode_returns

def save_video(frames, filename, fps=30):
    """Save frames as MP4 video"""
    if not frames or len(frames) == 0:
        print("No frames to save")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Create animation
    im = ax.imshow(frames[0])
    
    def animate(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), 
        interval=1000/fps, blit=True, repeat=False
    )
    
    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(filename, writer=writer)
        plt.close(fig)
        print(f"Video saved as {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Make sure ffmpeg is installed for video saving")
        plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate Mujoco RL agent")
    parser.add_argument(
        "--job", type=str, choices=["train", "eval"], default="train",
        help="Specify whether to run training or evaluation"
    )
    args = parser.parse_args()

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "PandaPickCubeCartesian",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": False, # always false for image-based envs
        "DEBUG": True,
    }

    if args.job == "train":
        rng = jax.random.PRNGKey(30)
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        final_train_state = out["runner_state"][0]
        save_checkpoint(final_train_state, step=config["TOTAL_TIMESTEPS"])
    
    elif args.job == "eval":
        print("\n" + "="*50)
        print("CREATING VIDEO ROLLOUTS")
        print("="*50)

        params = load_checkpoint("checkpoints/checkpoint_50000000.0_20250916_182415")  # Update with your checkpoint path
        
        videos, episode_returns = create_video_rollout(
            params, 
            config, 
            num_episodes=config.get("VIDEO_NUM_EPISODES", 3),
            max_steps=config.get("VIDEO_MAX_STEPS", 200)
        )
        
        # Save videos
        video_dir = config.get("VIDEO_OUTPUT_DIR", "./videos")
        os.makedirs(video_dir, exist_ok=True)
        
        for i, (frames, episode_return) in enumerate(zip(videos, episode_returns)):
            if frames:
                filename = f"{video_dir}/rollout_episode_{i+1}_return_{episode_return:.2f}.mp4"
                save_video(frames, filename)
        
        print(f"\nVideo rollouts saved in {video_dir}")
        print("Episode returns:", [f"{ret:.2f}" for ret in episode_returns])