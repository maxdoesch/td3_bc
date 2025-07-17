import os
import numpy as np
from typing import Dict, Optional, Tuple

import imageio.v3 as imageio
from gymnasium import Env, spaces

from dm_env import specs
from dm_control import suite
from dm_control.utils import io as resources
from dm_control.suite import common

from dmc_envs.utils import replace_green_bg, interpolate_bg


_DMC_ENVS_DIR = os.path.dirname(__file__)
_FILENAMES = [
    "./assets/materials.xml",
    "./assets/skybox.xml",
    "./assets/visual.xml",
]

ASSETS = {filename: resources.GetResource(os.path.join(_DMC_ENVS_DIR, filename)) for filename in _FILENAMES}


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int32(np.prod(s.shape))
        if type(s) is specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) is specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(Env):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        task_kwargs: Optional[Dict] = {},
        visualize_reward: bool = False,
        obs_type: str = "state",
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        environment_kwargs: Optional[Dict] = None,
        channels_first: bool = True,
    ):
        assert "random" in task_kwargs, "please specify a seed, for deterministic behaviour"
        assert obs_type in ["state", "pixels", "both"], "obs_type must be one of: state, pixels, both"
        self._obs_type = obs_type
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32)

        # create observation space
        if self._obs_type == "pixels":
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        elif self._obs_type == "state":
            self._observation_space = _spec_to_box(self._env.observation_spec().values(), np.float64)
        elif self._obs_type == "both":
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
                    "state": _spec_to_box(self._env.observation_spec().values(), np.float64),
                }
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values(), np.float64)

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get("random", 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs_info(self, time_step) -> Tuple:
        pixels = self.render(height=self._height, width=self._width, camera_id=self._camera_id).copy()
        pixels = pixels.transpose(2, 0, 1) if self._channels_first else pixels
        state = _flatten_obs(time_step.observation)

        if self._obs_type == "pixels":
            obs = pixels
            info = {'state': state}
        elif self._obs_type == "state":
            obs = state
            info = {'pixels': pixels}
        elif self._obs_type == "both":
            obs = {
                "pixels": pixels,
                "state": state,
            }
        return obs, info

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed=None):
        if seed is not None:
            seed = seed % 2**32

            np.random.seed(seed)
            self._true_action_space.seed(seed)
            self._norm_action_space.seed(seed)
            self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            terminated = time_step.last()
            if terminated:
                break
        obs, info = self._get_obs_info(time_step)
        extra["discount"] = time_step.discount
        extra.update(info)

        truncated = False  # dm_control has no time limits by default
        return obs, reward, terminated, truncated, extra

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs, info = self._get_obs_info(time_step)
        extra = {"internal_state": self._env.physics.get_state().copy(), "discount": time_step.discount}
        extra.update(info)
        return obs, extra

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


class DistractionDMCWrapper(DMCWrapper):
    """
    A wrapper for dm_control environments that implements distraction by playing videos in the background.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        task_kwargs: Optional[Dict] = None,
        visualize_reward: bool = False,
        obs_type: str = "pixels",
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        environment_kwargs: Optional[Dict] = None,
        channels_first: bool = True,
        video_dir: Optional[str] = None,
    ):
        super().__init__(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            obs_type=obs_type,
            height=height,
            width=width,
            camera_id=camera_id,
            frame_skip=frame_skip,
            environment_kwargs=environment_kwargs,
            channels_first=channels_first,
        )

        xml_model_string = common.read_model(domain_name + ".xml").decode("utf-8")

        to_replace = ["./common/skybox.xml", "./common/materials.xml", "./common/visual.xml"]
        for replace, replacement in zip(to_replace, _FILENAMES):
            xml_model_string = xml_model_string.replace(replace, replacement)

        self._env.physics.reload_from_xml_string(xml_model_string, assets=ASSETS)

        self._video_dir = video_dir if video_dir else os.path.join(os.path.dirname(__file__), "videos")
        self._video_paths = [
            os.path.join(self._video_dir, f) for f in os.listdir(self._video_dir) if f.endswith(".mp4")
        ]

        self._video_index = 0
        self._current_frame = 0
        self._data = None

    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video from provided filepath and return as numpy array"""
        video = imageio.imread(video_path, plugin="pyav")
        return np.moveaxis(video, -1, 1) if self._channels_first else video

    def reset(self, *, seed=None, options=None):
        self._video_index = np.random.randint(0, len(self._video_paths))
        self._data = self._load_video(self._video_paths[self._video_index])
        self._data = interpolate_bg(self._data, (self._height, self._width))

        self._current_frame = 0

        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._current_frame = (self._current_frame + 1) % self._data.shape[0]

        return super().step(action)

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        obs = super().render(mode=mode, height=height, width=width, camera_id=camera_id)

        return replace_green_bg(obs, self._data[self._current_frame])
