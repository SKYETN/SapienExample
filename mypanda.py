import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
@register_agent()
class MyPanda(BaseAgent):
    uid = "my_panda"
    urdf_path = "/home/shiqintong/Downloads/Wheelchair_urdf-main/Wheelchair_urdf/urdf/Wheelchair_urdf.urdf"
import mani_skill.envs
env = gym.make("EmptyEnv-v1", robot_uids="my_panda")

