import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from scipy.spatial.transform import Rotation as R
import mani_skill.envs
import argparse
from mani_skill.agents.controllers.pd_joint_vel import PDJointVelControllerConfig
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from copy import deepcopy
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from typing import List, Optional, Annotated, Union
import mplib
import trimesh
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
import sapien.physx as physx
from my_panda_motion_planner import MyPandaMotionPlanningSolver  # 新增


@register_agent()
class MyPanda(BaseAgent):
    uid = "my_panda"
    urdf_path = "/home/shiqintong/Downloads/wheelchair_description/urdf/inte.urdf"
    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[1.4, -1.0, 0.01], q=R.from_euler("z", 90, degrees=True).as_quat().tolist()),
            qpos=np.array([
                # arm: 7 joints
                -1.75, -0.5, 0.0, 1.0, 0.0, -1.5, 1.4, 
                # -1.75, -0.35, -3.3, 1.4, 3.4, -1.1, 1.4,
                # gripper: 6 joints
                0.04,  # finger_joint
                0.04, 0.04, 0.04, 0.04, 0.04  # mimic joints
            ])
        )
    )
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=5.0, dynamic_friction=5.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.03, min_patch_radius=0.01
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.03, min_patch_radius=0.01
            ),
            left_inner_finger=dict(material="gripper"),
            right_inner_finger=dict(material="gripper"),
        ),
    )
    def is_grasping(self, obj) -> bool:
        # 在 robot.links 中查找 name 为 left_inner_finger_pad 和 right_inner_finger_pad 的 link
        left_finger_pad = next(link for link in self.robot.links if link.get_name() == "left_inner_finger_pad")
        right_finger_pad = next(link for link in self.robot.links if link.get_name() == "right_inner_finger_pad")

        contacts = self.scene.get_contacts()

        left_contact = any(
            (c.actor0 == obj or c.actor1 == obj) and 
            (c.actor0 == left_finger_pad or c.actor1 == left_finger_pad)
            for c in contacts
        )
        right_contact = any(
            (c.actor0 == obj or c.actor1 == obj) and 
            (c.actor0 == right_finger_pad or c.actor1 == right_finger_pad)
            for c in contacts
        )

        return left_contact and right_contact


    @property
    def _controller_configs(self):
        arm_joint_names = [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7",
        ]
        gripper_joint_names = ["finger_joint"]

        arm_pd_joint_pos = PDJointPosControllerConfig(
            arm_joint_names,
            lower=[-3.14] * len(arm_joint_names),
            upper=[3.14] * len(arm_joint_names),
            stiffness=1000,
            damping=100,
            force_limit=100,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            arm_joint_names,
            lower=[-0.1] * len(arm_joint_names),
            upper=[0.1] * len(arm_joint_names),
            stiffness=1000,
            damping=100,
            force_limit=100,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            gripper_joint_names,
            lower=[0.00],
            upper=[0.80],
            stiffness=1000,
            damping=100,
            force_limit=100,
        )

        return deepcopy({
            "pd_joint_delta_pos": {
                "arm": arm_pd_joint_delta_pos,
                "gripper": gripper_pd_joint_pos,
            },
            "pd_joint_pos": {
                "arm": arm_pd_joint_pos,
                "gripper": gripper_pd_joint_pos,
            },
        })



@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "Empty-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "my_panda"
    """The robot to use. Robot setups supported for teleop in this script are panda and panda_stick"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

    keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None
    """Name of keyframe to view"""
    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """Pause viewer on load"""


def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        robot_uids=args.robot_uid,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(
        seed=seed,
    )

    # 初始化位姿
    kf = None
    if env.agent.keyframes:
        kf_name = args.keyframe or next(iter(env.agent.keyframes))
        kf = env.agent.keyframes[kf_name]
        env.agent.robot.set_pose(kf.pose)
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        print(f"📌 Viewing keyframe: {kf_name}")

    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    viewer = env.render()
    viewer.paused = args.pause

    # ✅ 设置 robot base pose
    from scipy.spatial.transform import Rotation as R
    base_pose = sapien.Pose(
        [1.4, -3.5, 0.01],  # 可按实际情况调整
        R.from_euler("z", 90, degrees=True).as_quat().tolist()
    )


    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(
                seed=seed,
            )
            continue
        elif code == "restart":
            env.reset(
                seed=seed,
                options=dict(
                    save_trajectory=False,
                )
            )
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            robot_uids=args.robot_uid,
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env

def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = False

    if env.unwrapped.robot_uids == "my_panda":  # 新增支持 MyPanda
        robot_has_gripper = True
        planner = MyPandaMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    else:
        raise ValueError(f"Unsupported robot: {env.unwrapped.robot_uids}")

    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_panda_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "tool_frame")._objs[0].entity)  # 修改为 tool_frame
    select_panda_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:
        transform_window.enabled = True
        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the panda hand up
            j: move the panda hand down
            arrow_keys: move the panda hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            print("gripper control")
            if gripper_open:
                print("gripper open - close")
                gripper_open = False
                _, reward, _, _, info = planner.close_gripper()
            else:
                print("gripper close - open")
                gripper_open = True
                _, reward, _, _, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        if execute_current_pose:
            # # 替换姿态：设置为水平向量（绕 x/y/z 轴的旋转角度）
            # target_pose = transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1])
            # r = R.from_euler('xyz', [0, 0, 0])  # 或 [0, 0, np.pi/2] 具体看你想让末端平行哪条轴
            # target_pose.q = r.as_quat()
            # result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
            result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _, _, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1:
                    print("Plan failed")
                else:
                    print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False





if __name__ == "__main__":
    main(parse_args())
