import functools
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from flax import struct
from mujoco_playground import wrapper
from mujoco_playground import manipulation
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper

import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray 
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray 
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray

class MujocoWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class LogWrapper(MujocoWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        n = obs.shape[0]
        zeros_f = jnp.zeros((n,), dtype=jnp.float32)
        zeros_i = jnp.zeros((n,), dtype=jnp.int32)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=zeros_f,
            episode_lengths=zeros_i,
            returned_episode_returns=zeros_f,
            returned_episode_lengths=zeros_i,
            timestep=zeros_i,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

class MJXWrapper:
    def __init__(self, env_name, num_envs=2048, train=True):
        # Load the playground environment
        self.num_envs = num_envs
        env_cfg = manipulation.get_default_config(env_name)
        episode_length = int(4 / env_cfg.ctrl_dt)

        height, width = 64, 64
        config_overrides = {
            "episode_length": episode_length,
            "vision": True,
            "obs_noise.brightness": [0.75, 2.0],
            "vision_config.use_rasterizer": False,
            "vision_config.render_batch_size": num_envs,
            "vision_config.render_width": 64,
            "vision_config.render_height": 64,
            "box_init_range": 0.1, # +- 10 cm
            "action_history_length": 5,
            "success_threshold": 0.03
        }

        env = manipulation.load(env_name, config=env_cfg, config_overrides=config_overrides)
        # randomization_fn = functools.partial(randomize.domain_randomize,
        #                                         num_worlds=)
        if train:
            env = wrapper.wrap_for_brax_training(
                env,
                vision=True,
                num_vision_envs=num_envs,
                episode_length=episode_length,
                action_repeat=1,
                # randomization_fn=randomization_fn
            )
        # env = AutoResetWrapper(env)
        self._env = env

        self.observation_shape = (height, width, 3)
    
    def _get_obs(self, state):
        if "pixels/view_0" not in state.obs:
            raise ValueError("Environment does not have pixel observations.")
        
        return state.obs["pixels/view_0"]
        
    def reset(self, rng, params=None):
        state = self._env.reset(rng)
        return self._get_obs(state), state
    
    def step(self, rng, state, action, params=None):
        state = self._env.step(state, action)
        return self._get_obs(state), state, state.reward, state.done > 0.5, {}
    
    def observation_space(self, params=None):
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_shape,
        )
    
    def action_space(self, params=None):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )