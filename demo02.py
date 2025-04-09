import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from scipy.spatial.transform import Rotation as R
import mani_skill.envs
import argparse

from mani_skill.agents.controllers.pd_joint_vel import PDJointVelControllerConfig
import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
from copy import deepcopy



import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union


import mplib
import numpy as np
import sapien
import trimesh

import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
import sapien.physx as physx
OPEN = 1
CLOSED = -1


class MyPandaMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf="/home/shiqintong/Downloads/wheelchair_description/urdf/inte.urdf",
            srdf="/home/shiqintong/Downloads/wheelchair_description/urdf/inte.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="tool_frame",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass

from transforms3d import quaternions


def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual





def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="The id of the robot to place in the environment")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos", help="The control mode to use. Note that for new robots being implemented if the _controller_configs is not implemented in the selected robot, we by default provide two default controllers, 'pd_joint_pos' and 'pd_joint_delta_pos' ")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--keyframe-actions", action="store_true", help="Whether to use the selected keyframe to set joint targets to try and hold the robot in its position")
    parser.add_argument("--random-actions", action="store_true", help="Whether to sample random actions to control the agent. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--none-actions", action="store_true", help="If set, then the scene and rendering will update each timestep but no joints will be controlled via code. You can use this to control the robot freely via the GUI.")
    parser.add_argument("--zero-actions", action="store_true", help="Whether to send zero actions to the robot. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--sim-freq", type=int, default=100, help="Simulation frequency")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    
    return args

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "Empty-v1"
    """Environment ID (e.g. PushCube-v1, Empty-v1)"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "panda"
    """Robot UID(s). Comma-separated or single string. Default: panda"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Simulation backend: auto, cpu, gpu"""

    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode (e.g. pd_joint_pos, pd_joint_vel, etc.)"""

    keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None
    """Name of keyframe to view"""

    shader: str = "default"
    """Shader used for rendering (default, rt, rt-fast)"""

    keyframe_actions: bool = False
    """Use keyframe to control robot pose"""

    random_actions: bool = False
    """Send random actions each step"""

    none_actions: bool = False
    """Send no actions (manual GUI only)"""

    zero_actions: bool = False
    """Send zero actions"""

    sim_freq: int = 100
    """Simulation frequency"""

    control_freq: int = 20
    """Control frequency"""

    obs_mode: str = "none"
    """Observation mode"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    render_mode: str = "human"
    """Render mode (rgb_array, human, etc.)"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """Pause viewer on load"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    quiet: bool = False
    """Disable verbose output"""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Random seed or list of seeds"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments"""
    plan_actions: bool = False 
@register_agent()
class MyPanda(BaseAgent):
    uid = "my_panda"
    urdf_path = "/home/shiqintong/Downloads/wheelchair_description/urdf/inte.urdf"
    keyframes = dict(
        standing=Keyframe(
            # notice how we set the z position to be above 0, so the robot is not intersecting the ground
            pose=sapien.Pose(p=[1.4, -1.5, 0.01], q=R.from_euler("z", 90, degrees=True).as_quat().tolist()),
            qpos=np.array([
                # arm: 7 joints
                -1.75, -0.35, -3.3, 1.4, 3.4, -1.1, 1.4,0.0, 0.0, 0.0, 0.0,
                # gripper: 6 joints
                0.04,  # finger_joint
                0.04, 0.04, 0.04, 0.04, 0.04  # mimic joints
            ])
        )
    )
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_inner_finger=dict(material="gripper"),
            right_inner_finger=dict(material="gripper"),
        ),
    )
    
    @property
    def _controller_configs(self):
        arm_joint_names = [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7",
        ]
        gripper_joint_names = ["finger_joint"]
                # ✅ 添加 wheel joints
        wheel_joint_names = [
            "joint_LF01", "joint_LB01", "joint_RF01", "joint_RB01",
        ]

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
            lower=[-0.01],
            upper=[0.04],
            stiffness=1000,
            damping=100,
            force_limit=100,
        )
        wheel_pd_joint_pos = PDJointPosControllerConfig(
            wheel_joint_names,
            lower=[-100] * len(wheel_joint_names),
            upper=[100] * len(wheel_joint_names),
            stiffness=1000,
            damping=50,
            force_limit=50,
            normalize_action=False,
        )

        return deepcopy({
            "pd_joint_delta_pos": {
                "arm": arm_pd_joint_delta_pos,
                "gripper": gripper_pd_joint_pos,
                "wheels": wheel_pd_joint_pos,  
            },
            "pd_joint_pos": {
                "arm": arm_pd_joint_pos,
                "gripper": gripper_pd_joint_pos,
                "wheels": wheel_pd_joint_pos,  
            },
        })


    # wheel_joint_names = [
    #     "joint_LF01",
    #     "joint_LB01",
    #     "joint_RF01",
    #     "joint_RB01"
    # ]
    #     # 设定控制参数（可根据实际需要修改）
    # wheel_stiffness = 1000
    # wheel_damping = 100
    # wheel_force_limit = 100
    
    
    
# ✅ 主函数
def main():
    args = tyro.cli(Args)

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uids,
        sensor_configs={"shader_pack": args.shader},
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={"shader_pack": args.shader},
        render_mode=args.render_mode,
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )

    env.reset(seed=args.seed or 0)
    env: BaseEnv = env.unwrapped

    print(f"✅ Selected robot: {args.robot_uids}, control mode: {args.control_mode}")
    print(f"🔑 Available keyframes: {list(env.agent.keyframes.keys())}")
    import sapien
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building import URDFLoader
    loader = URDFLoader()
    loader.set_scene(ManiSkillScene())
    robot = loader.load("/home/shiqintong/Downloads/wheelchair_description/urdf/inte.urdf")
    print(robot.active_joints_map.keys())







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

    planner = None
    if args.plan_actions:
        from transforms3d.euler import euler2quat
        from transforms3d.euler import mat2euler

        planner = MyPandaMotionPlanningSolver(
            env,
            debug=False,
            vis=True,
            base_pose=env.agent.robot.pose, # 你也可指定别的
            visualize_target_grasp_pose=False,
            print_env_info=True,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    arm_qpos = env.agent.robot.get_qpos()[0, :7].cpu().numpy()
    gripper_state = 0.8
        # 初始化轮子状态
    wheel_qpos = {
        "joint_LF01": 0.0,
        "joint_LB01": 0.0,
        "joint_RF01": 0.0,
        "joint_RB01": 0.0,
    }

    # 获取轮子 joint index
    wheel_joint_ids = {
        name: env.agent.robot.active_joints_map[name].index
        for name in wheel_qpos
    }
    while True:
        viewer = env.render()

        if args.random_actions:
            # 控制机械臂
            if viewer.window.key_down("b"):
                arm_qpos[1] += 0.05
            if viewer.window.key_down("n"):
                arm_qpos[1] -= 0.05
            if viewer.window.key_down("z"):
                arm_qpos[2] += 0.05
            if viewer.window.key_down("x"):
                arm_qpos[2] -= 0.05
            if viewer.window.key_down("c"):
                arm_qpos[3] += 0.05
            if viewer.window.key_down("v"):
                arm_qpos[3] -= 0.05
            if viewer.window.key_down("m"):
                arm_qpos[4] += 0.05
            if viewer.window.key_down("f"):
                arm_qpos[4] -= 0.05
            if viewer.window.key_down("g"):
                arm_qpos[5] += 0.05
            if viewer.window.key_down("h"):
                arm_qpos[5] -= 0.05
            if viewer.window.key_down("r"):
                arm_qpos[6] += 0.05
            if viewer.window.key_down("t"):
                arm_qpos[6] -= 0.05
            if viewer.window.key_down("up"):
                arm_qpos[0] += 0.05
            if viewer.window.key_down("down"):
                arm_qpos[0] -= 0.05

            # 控制夹爪
            if viewer.window.key_down("o"):
                gripper_state = 0.1
            if viewer.window.key_down("p"):
                gripper_state = 0.0

            # 控制轮子
            delta = 0.1
            if viewer.window.key_down("i"):  # 前进
                for k in wheel_qpos:
                    wheel_qpos[k] += delta
            if viewer.window.key_down("k"):  # 后退
                for k in wheel_qpos:
                    wheel_qpos[k] -= delta
            if viewer.window.key_down("j"):  # 左转
                wheel_qpos["joint_LF01"] -= delta
                wheel_qpos["joint_LB01"] -= delta
                wheel_qpos["joint_RF01"] += delta
                wheel_qpos["joint_RB01"] += delta
            if viewer.window.key_down("l"):  # 右转
                wheel_qpos["joint_LF01"] += delta
                wheel_qpos["joint_LB01"] += delta
                wheel_qpos["joint_RF01"] -= delta
                wheel_qpos["joint_RB01"] -= delta

            # 构造总的 action (7 + 1 + 4 = 12)
            action = np.hstack([
                arm_qpos,               # 7 joints
                [gripper_state],        # 1 gripper joint
                [wheel_qpos["joint_LF01"]],
                [wheel_qpos["joint_LB01"]],
                [wheel_qpos["joint_RF01"]],
                [wheel_qpos["joint_RB01"]],
            ])[None, :]  # shape: (1, 12)
            # else:
            #     print("⚠️ 当前控制器不是 DictController，请手动拼接 action！")
            #     action = np.hstack([
            #         arm_qpos,
            #         gripper_state,
            #         wheel_qpos["joint_LF01"],
            #         wheel_qpos["joint_LB01"],
            #         wheel_qpos["joint_RF01"],
            #         wheel_qpos["joint_RB01"],
            #     ])[None, :]

            env.step(action)
            print("Current qpos:", np.round(arm_qpos, 3))

        elif args.none_actions:
            env.step(None)
        elif args.zero_actions:
            env.step(np.zeros_like(env.action_space.sample()))
        elif args.keyframe_actions:
            if kf is not None:
                if isinstance(env.agent.controller, DictController):
                    env.step(env.agent.controller.from_qpos(kf.qpos))
                else:
                    env.step(kf.qpos)
        elif args.plan_actions:
            target_pose = sapien.Pose([0.0, 0.0, 0.0])
            result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
            if result != -1:
                planner.follow_path(result)

    # while True:
    #         viewer = env.render()

    #         if args.random_actions:
    #             # 控制机械臂
    #             if viewer.window.key_down("b"):
    #                 arm_qpos[1] += 0.05
    #             if viewer.window.key_down("n"):
    #                 arm_qpos[1] -= 0.05

    #             # 控制夹爪
    #             if viewer.window.key_down("o"):
    #                 gripper_state = 0.1
    #             if viewer.window.key_down("p"):
    #                 gripper_state = 0.0

    #             # 控制轮子
    #             delta = 0.1
    #             if viewer.window.key_down("i"):
    #                 for k in wheel_qpos:
    #                     wheel_qpos[k] += delta
    #             if viewer.window.key_down("k"):
    #                 for k in wheel_qpos:
    #                     wheel_qpos[k] -= delta
    #             if viewer.window.key_down("j"):
    #                 wheel_qpos["joint_LF01"] -= delta
    #                 wheel_qpos["joint_LB01"] -= delta
    #                 wheel_qpos["joint_RF01"] += delta
    #                 wheel_qpos["joint_RB01"] += delta
    #             if viewer.window.key_down("l"):
    #                 wheel_qpos["joint_LF01"] += delta
    #                 wheel_qpos["joint_LB01"] += delta
    #                 wheel_qpos["joint_RF01"] -= delta
    #                 wheel_qpos["joint_RB01"] -= delta

    #             # 构造总的 action
    #             total_qpos = np.zeros(env.agent.robot.dof)

    #             # 机械臂 7 joints + gripper
    #             total_qpos[:7] = arm_qpos
    #             total_qpos[7] = gripper_state

    #             # 设置轮子 joint 的角度
    #             for name, value in wheel_qpos.items():
    #                 index = wheel_joint_ids[name]
    #                 total_qpos[index] = value

    #             action = total_qpos[None, :]  # (1, DOF)
    #             env.step(action)

    #         elif args.none_actions:
    #             env.step(None)
    #         elif args.zero_actions:
    #             env.step(np.zeros_like(env.action_space.sample()))
    #         elif args.keyframe_actions:
    #             if kf is not None:
    #                 if isinstance(env.agent.controller, DictController):
    #                     env.step(env.agent.controller.from_qpos(kf.qpos))
    #                 else:
    #                     env.step(kf.qpos)
    #         elif args.plan_actions:
    #             # 示例路径规划
    #             target_pose = sapien.Pose([0.0, 0.0, 0.0])
    #             result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
    #             if result != -1:
    #                 planner.follow_path(result)

    # # ✅ 主循环
    # while True:
    #     if args.random_actions:
    #         # full_qpos = env.agent.robot.get_qpos()[0].cpu().numpy()

    #         # # 只取前 7 个 arm joints
    #         # arm_qpos = full_qpos[:7]

    #         # 键盘控制 joint_3 上下（索引 2）
    #         if viewer.window.key_down("b"):
    #             arm_qpos[1] += 0.05  # joint_3 向上
    #         if viewer.window.key_down("n"):
    #             arm_qpos[1] -= 0.05  # joint_3 向下


    #         if viewer.window.key_down("o"):
    #             gripper_state = 0.1  # fully open
    #         if viewer.window.key_down("p"):
    #             gripper_state = 0.0  # fully closed

    #         # 构造 action
    #         if env.control_mode == "pd_joint_pos":
    #             action = np.hstack([arm_qpos, gripper_state])[None, :]  # (1, 8)
    #         elif env.control_mode == "pd_joint_pos_vel":
    #             qvel = np.zeros_like(arm_qpos)
    #             action = np.hstack([arm_qpos, qvel, gripper_state])[None, :]





    #         else:
    #             print("⚠️ 当前控制模式未处理，使用随机动作代替")
    #             action = env.action_space.sample()

    #         env.step(action)
    #         # env.step(env.action_space.sample())

    #         # env.step(env.action_space.sample())
    #     elif args.none_actions:
    #         env.step(None)
    #     elif args.zero_actions:
    #         env.step(np.zeros_like(env.action_space.sample()))
    #     elif args.keyframe_actions:
    #         if kf is not None:
    #             if isinstance(env.agent.controller, DictController):
    #                 env.step(env.agent.controller.from_qpos(kf.qpos))
    #             else:
    #                 env.step(kf.qpos)
    #     elif args.plan_actions:
    #         # 例如：规划一个目标末端位姿
    #         target_pose = sapien.Pose([0.0, 0.0, 0.0])  # 随便写
    #         result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
    #         if result == -1:
    #             print("Plan failed!")
    #         else:
    #             planner.follow_path(result)

    #         # 这里演示一次就够，如果你要每帧做不同的规划，可以自己改逻辑

    #     viewer = env.render()

if __name__ == "__main__":
    main()

