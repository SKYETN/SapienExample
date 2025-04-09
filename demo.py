import gymnasium as gym

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mshab.envs
from mshab.envs.planner import plan_data_from_file


task = "set_table" # "tidy_house", "prepare_groceries", or "set_table"
subtask = "open"    # "sequential", "pick", "place", "open", "close"
                    # NOTE: sequential loads the full task, e.g pick -> place -> ...
                    #     while pick, place, etc only simulate a single subtask each episode
split = "val"     # "train", "val"


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
)
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

env = gym.make(
    f"{subtask.capitalize()}SubtaskTrain-v0",
    # Simulation args
    num_envs=252,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
    obs_mode="rgbd",
    sim_backend="gpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    # Rendering args
    reward_mode="normalized_dense",
    render_mode="rgb_array",
    shader_dir="minimal",
    # TimeLimit args
    max_episode_steps=200,
    # SequentialTask args
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # optional: additional env_kwargs
)

# add env wrappers here

venv = ManiSkillVectorEnv(
    env,
    max_episode_steps=1000,  # set manually based on task
    ignore_terminations=True,  # set to False for partial resets
)

# add vector env wrappers here

obs, info = venv.reset()
