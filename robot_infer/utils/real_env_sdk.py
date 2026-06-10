import collections
import os
import sys
import time

import dm_env
import numpy as np

sys.path.append("../")
sys.path.append("../../")

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
build_dir = os.path.join(root, "lib")
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from lib_h1_sdk_python import ArmAction
from lib_h1_sdk_python import EtherCAT_Motor_Index
from lib_h1_sdk_python import H1Robot
from lib_h1_sdk_python import MotorControl
from lib_h1_sdk_python import MotorControlMode
from lib_h1_sdk_python import MotorInformation

from .camera_sdk import ImageRecorder


DEFAULT_CAMERA_NAMES = ["cam_left_wrist", "cam_high", "cam_right_wrist"]


def camera_aliases(camera_name: str) -> list[str]:
    return [
        camera_name,
        f"rs/{camera_name}",
        f"rs.{camera_name}",
        f"rs_{camera_name}",
    ]


def camera_sources(camera_names: list[str]) -> list[str]:
    sources = []
    for camera_name in camera_names:
        for alias in camera_aliases(camera_name):
            if alias not in sources:
                sources.append(alias)
    return sources


def normalize_camera_images(images: dict, camera_names: list[str]) -> dict:
    normalized = dict(images)
    rs_images = images.get("rs")

    for camera_name in camera_names:
        if camera_name in normalized:
            continue

        if isinstance(rs_images, dict) and camera_name in rs_images:
            normalized[camera_name] = rs_images[camera_name]
            continue

        for alias in camera_aliases(camera_name):
            if alias in images:
                normalized[camera_name] = images[alias]
                break

    return normalized


class Real_Env:
    def __init__(self, camera_names=None):
        self.camera_names = camera_names or DEFAULT_CAMERA_NAMES
        self.gripper_state = np.zeros(2)
        self.waist_state = np.zeros(2)
        self.left_eef_state = np.zeros(6)
        self.right_eef_state = np.zeros(6)
        self.head_target = np.zeros(6)
        self.T_l_init2base = np.array(
            [
                [0.99595273, 0.0, -0.08987855, 0.39210682],
                [0.0, 1.0, 0.0, 0.18292719],
                [0.08987855, 0.0, 0.99595273, 0.19168143],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.T_r_init2base = np.array(
            [
                [0.99595273, 0.0, -0.08987855, 0.39233398],
                [0.0, 1.0, 0.0, -0.18307313],
                [0.08987855, 0.0, 0.99595273, 0.19178583],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.robot = H1Robot()
        self.image_recorder = ImageRecorder(camera_sources(self.camera_names))

        self.arm_motor_index = [eval(f"EtherCAT_Motor_Index.MOTOR_LEFT_ARM_{idx + 1}") for idx in range(8)] + [
            eval(f"EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_{idx + 1}") for idx in range(8)
        ]
        self.waist_motor_index = [
            EtherCAT_Motor_Index.MOTOR_LIFT,
            EtherCAT_Motor_Index.MOTOR_WAIST_DOWN,
            EtherCAT_Motor_Index.MOTOR_WAIST_UP,
        ]
        self.head_motor_index = [EtherCAT_Motor_Index.MOTOR_HEAD_DOWN, EtherCAT_Motor_Index.MOTOR_HEAD_UP]
        self.motor_index = self.arm_motor_index + self.waist_motor_index + self.head_motor_index
        self.motor_num = len(self.motor_index)

        self.right_arm_target_pos_cache = None
        self.robot_state_cfg = {
            "still_range_threshold": 0.05,
            "still_timeout_threshold": 1.0,
            "acc_max_threshold": 0.1,
        }

        max_retry = 10
        count = 0
        while count < max_retry and not self.robot.isRobotConnected():
            try:
                self.robot.robot_connect()
            except Exception:
                print("connect failed, trying to connect again.")
                time.sleep(1)
                count += 1

        if not self.robot.switchControlMode(MotorControlMode.GRAVITY_COMPENSATION_LEVEL):
            print("switch mode failed")
            return
        print("mode =", self.robot.getCurrentMode())

        if not self.robot.robot_init():
            print("robot_init failed")
            return
        print("robot initialized")

    def step_joint(self, action):
        obs = self.get_observation()
        self._set_joint_action(action)
        return obs

    def _set_joint_action(self, action):
        assert len(action) == self.motor_num
        for idx in range(self.motor_num):
            self.send_motor(self.motor_index[idx], pos=action[idx])

    def _set_head_joint(self, action):
        head_motor_num = len(self.head_motor_index)
        for idx in range(head_motor_num):
            self.send_motor(self.head_motor_index[idx], pos=action[idx])

    def publish_waist_joint(self, action):
        assert len(action) == len(self.waist_motor_index)
        for i, idx in enumerate(self.waist_motor_index):
            self.send_motor(idx, pos=action[i])

    def move_to_init_pose(self):
        target_joint_positions = np.zeros(self.motor_num)
        current_joint_positions = self.get_joint_position()

        print("Moving to initial pose...")
        move_time = 2.0
        steps = 1000
        height_motor_index = 16
        target_joint_positions[height_motor_index] = current_joint_positions[height_motor_index]

        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_positions = (1 - alpha) * current_joint_positions + alpha * target_joint_positions
            self._set_joint_action(interpolated_positions)
            time.sleep(move_time / steps)

        print("Reached initial pose.")

    def move_to_deinit_pose(self):
        if not self.robot.robot_deinit():
            print("robot_deinit failed")
            return
        print("robot deinitialized")

    def move_to_target_joint(self, action):
        target_joint_positions = action
        current_joint_positions = self.get_joint_position()

        print("Moving to target joint pose...")
        move_time = 2.0
        steps = 1000

        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_positions = (1 - alpha) * current_joint_positions + alpha * target_joint_positions
            self._set_joint_action(interpolated_positions)
            time.sleep(move_time / steps)

        print("Reached target joint pose.")

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_joint_position()
        obs["qpos"] = np.pad(obs["qpos"], (0, 2), mode="constant", constant_values=0)
        obs["images"] = normalize_camera_images(self.image_recorder.get_images(), self.camera_names)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=obs,
        )

    def get_robot_is_normal(self) -> bool:
        results = self.robot.getArmTargetState(ArmAction.RIGHT_ARM)
        ret, pos = results[0], results[1].position
        if not ret:
            return False
        if self.right_arm_target_pos_cache is None:
            self.right_arm_target_pos_cache = np.tile(pos, (20, 1))
        self.right_arm_target_pos_cache = np.roll(self.right_arm_target_pos_cache, -1, axis=0)
        self.right_arm_target_pos_cache[-1] = pos

        acc = self.right_arm_target_pos_cache[-1] - self.right_arm_target_pos_cache[-2]
        if np.linalg.norm(acc) > self.robot_state_cfg["acc_max_threshold"]:
            return False

        pos_std = np.linalg.norm(np.std(self.right_arm_target_pos_cache, axis=0))
        if pos_std < self.robot_state_cfg["still_range_threshold"]:
            return False

        return True

    def get_joint_position(self):
        joint_position = np.zeros(self.motor_num)
        for idx in range(self.motor_num):
            _, st = self.get_state(self.motor_index[idx])
            joint_position[idx] = st.Position_Actual
        return joint_position

    def get_gripperstate(self):
        try:
            success_left, left_gripper_info = self.robot.getGripperState(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8)
            success_right, right_gripper_info = self.robot.getGripperState(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8)

            if success_left:
                self.gripper_state[0] = left_gripper_info.Position_Actual
            if success_right:
                self.gripper_state[1] = right_gripper_info.Position_Actual
        except Exception as exc:
            print(f"Failed to get gripper state: {exc}")

    def get_waiststate(self):
        try:
            success_height, height_info = self.robot.getWaistState(EtherCAT_Motor_Index.MOTOR_LIFT)
            success_pitch, pitch_info = self.robot.getWaistState(EtherCAT_Motor_Index.MOTOR_WAIST_DOWN)
            success_yaw, yaw_info = self.robot.getWaistState(EtherCAT_Motor_Index.MOTOR_WAIST_UP)

            self.head_target = np.zeros(6)
            if success_height:
                self.head_target[2] = height_info.Position_Actual
            if success_pitch:
                self.head_target[4] = pitch_info.Position_Actual
            if success_yaw:
                self.head_target[5] = yaw_info.Position_Actual
        except Exception as exc:
            print(f"Failed to get waist state: {exc}")

    def get_images(self):
        return normalize_camera_images(self.image_recorder.get_images(), self.camera_names)

    def reset(self):
        return self.get_observation()

    def send_motor(self, idx, pos=None, speed=None, torque=None):
        """Send a low-level motor command. Position is used for joints; speed is used for chassis wheels."""
        mc = MotorControl()
        if pos is not None:
            mc.Position = pos
        if speed is not None:
            mc.Speed = speed
        if torque is not None:
            mc.Torque = torque
        self.robot.setMotorControl_low(idx, mc)

    def get_state(self, idx):
        """Read one motor state while supporting several SDK binding signatures."""
        if not hasattr(self.robot, "getMotorState"):
            return None
        m = self.robot.getMotorState
        try:
            ret = m(idx)
            if isinstance(ret, MotorInformation):
                return ret
            if isinstance(ret, (list, tuple)):
                try:
                    return ret[idx]
                except Exception:
                    pass
            return ret
        except TypeError:
            pass
        try:
            out = MotorInformation()
            ok = m(idx, out)
            if isinstance(ok, MotorInformation):
                return ok
            return out
        except TypeError:
            pass
        try:
            ret = m()
            if isinstance(ret, (list, tuple)):
                return ret[idx]
            return ret
        except TypeError:
            return None


def make_real_env(camera_names):
    env = Real_Env(camera_names)
    time.sleep(1)
    return env

