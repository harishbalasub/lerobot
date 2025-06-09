# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )

from ultralytics import YOLO
import cv2
import numpy as np

import matplotlib.pyplot as plt






# 4. Extract bboxes and store in observation
def get_bbox(boxes, names, label, prev_bbox):

    for box in boxes:
        if names[int(box.cls)] == label:            
            return box.xyxy[0].cpu().numpy()
    return prev_bbox  # Return a zero array if no box is found

def is_bbox_inside(inner, outer, threshold=0.8):
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))  # avoid div by 0
    return (inter_area / inner_area) >= threshold

def bbox_inside_prcnt(inner, outer):
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))  # avoid div by 0
    return (inter_area / inner_area) 

def compute_yolo_reward(pick_bbox, place_bbox):
    return float(is_bbox_inside(pick_bbox, place_bbox))

@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    # Policy is "iql_act"
    yolo_model = YOLO("yolotrain/runs/detect/train6/weights/best.pt")
    timestamp = 0
    bbox_frequency = 1
    bbox_update = 0
    
    #cmd_list = [ 0, 1, 0, 1, 2]
    #pick_list = [ "crinkle", "crinkle", "scrub", "scrub",""]
    #place_list = [ "", "tray", "", "bowl", ""]
    
    cmd_list = [ 0, 1, 2]
    pick_list = [ "scrub", "scrub", ""]
    place_list = [ "", "tray", ""]
    
    total_cmds = len(cmd_list)
    cmd_index = 0
    cmd_id = cmd_list[cmd_index] # pick 
    goal_pick_label = pick_list[cmd_index]  # "scrub"
    goal_place_label = place_list[cmd_index]  # "tray"
    pick_bbox = np.zeros(4)
    place_bbox = np.zeros(4)
    gripper_bbox = np.zeros(4)  # Initialize gripper bbox
    save_once = False  # To save the first image only once
    start_episode_t = time.perf_counter()
    wipe_duration = 0  # Initialize wipe duration counter
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
            #print(f"Teleoperation observation: {observation}")
            #if action is not None and int(timestamp) % 5 == 0:
            #    print(f"Teleoperation action: {action}")
        else:
            observation = robot.capture_observation()
            action = None


            
            if getattr(policy, "name", None) == "iql_act" or getattr(policy, "name", None) == "cql_act" or getattr(policy, "name", None) == "iql_pi0" or getattr(policy, "name", None) == "act_cond":
                if bbox_update % bbox_frequency == 0:
                    # 1. Extract image from observation
                    img_tensor = observation["observation.images.laptop"]  # shape (3, H, W)
                
                    # 2. Convert to OpenCV format
                    #img = img_tensor.permute(1, 2, 0).cpu().numpy()  # shape (H, W, 3)
                    img = img_tensor.clone().detach().cpu().numpy()
                    #img = img_tensor.permute(1, 2, 0).clone().detach().cpu().numpy()  # shape (H, W, 3)
                    #img = (img * 255).astype(np.uint8)    
                    #print(img.mean(), img.std())
                    #img = (img * 255).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR for OpenCV
                    
                    # 3. Run YOLO
                    #print(img.shape)
                    results = yolo_model.predict(img_rgb, imgsz=640, conf=0.25, iou=0.25, verbose=False)[0]
                    boxes = results.boxes
                    names = yolo_model.names


                    #use matplotlib to write the img
                    if bbox_update % 90 == 0 and not save_once:
                        plt.imshow(img)
                        plt.axis('off')
                        plt.savefig("img.png")
                        annotated_img = results.plot()  # Visualize YOLO detections
                        plt.imshow(annotated_img)
                        plt.axis('off')
                        plt.savefig("yolo_output.png")
                        print(f"YOLO class names: {names}")
                        print(f"Detected boxes: {[(names[int(box.cls)], box.conf.item(), box.xyxy[0].tolist()) for box in boxes]}")
                        print(f"Number of boxes: {len(boxes)}")   
                        
                        print(f"cmd_id {cmd_id} goal_pick_label: {goal_pick_label} in {pick_bbox}, goal_place_label: {goal_place_label} in {place_bbox}, gripper_bbox: {gripper_bbox}")                 
                        print(f"gripper inside pick {bbox_inside_prcnt(gripper_bbox, pick_bbox)} pick inside gripper {bbox_inside_prcnt(pick_bbox, gripper_bbox)}")
                        print(f"pick inside place {bbox_inside_prcnt(pick_bbox, place_bbox)}")
                        

 
            
                    pick_orientation = [0, 0, 0, 0]  # Placeholder for pick orientation
                    place_orientation = [0, 0, 0, 0]  # Placeholder for place orientation
                    
                    if cmd_id == 0 or cmd_id == 3:
                        pick_bbox = get_bbox(boxes, names, goal_pick_label, pick_bbox)
                        place_bbox = [0, 0, 0, 0]  # Reset place_bbox
                    elif cmd_id == 1:
                        pick_bbox = get_bbox(boxes, names, goal_pick_label, pick_bbox)
                        place_bbox = get_bbox(boxes, names, goal_place_label, place_bbox)
                    elif cmd_id == 2 or cmd_id == 4:
                        pick_bbox = [0, 0, 0, 0]  # Reset pick_bbox
                        place_bbox = [0, 0, 0, 0]  # Reset place_bbox     
                    else:
                        assert False, "unsupported cmd_id in control_loop"

                    gripper_bbox = get_bbox(boxes, names, "gripper", gripper_bbox)
                bbox_update += 1
                if 1:
                    observation["goal.pick_bbox"] = torch.tensor(pick_bbox, dtype=torch.float32)
                    observation["goal.place_bbox"] = torch.tensor(place_bbox, dtype=torch.float32)
                    cmd_one_hot = torch.zeros(32, dtype=torch.float32)
                    cmd_one_hot[cmd_id] = 1.0
                    observation["cmd_one_hot"] = cmd_one_hot
                    #observation["goal.pick_orientation"] = torch.tensor(pick_orientation, dtype=torch.float32)
                    #observation["goal.place_orientation"] = torch.tensor(place_orientation, dtype=torch.float32)
                


            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                if bbox_update % 90 == 0:
                    print(f"Action sent: {action}")                
                action = {"action": action}


            if getattr(policy, "name", None) == "act_cond" or getattr(policy, "name", None) == "cql_act" :
                if cmd_id == 0 or cmd_id == 3:
                    # If the command is pick, we need to reset the place bbox                    
                    if action["action"][1] > 150 and action["action"][2] > 100 and \
                               (is_bbox_inside(gripper_bbox, pick_bbox, threshold=0.2) \
                               or is_bbox_inside(pick_bbox, gripper_bbox, threshold=0.2)):
                        # If the gripper is inside the pick or place bbox, we reset the pick bbox
                        print("##########PICK SUCCESS##########")
                        cmd_index += 1
                        if cmd_index >= total_cmds:
                            break
                        cmd_id = cmd_list[cmd_index]
                        wipe_duration = timestamp  if cmd_id == 4 else wipe_duration
                        goal_pick_label = pick_list[cmd_index]  # "scrub"
                        goal_place_label = place_list[cmd_index]  # "tray"
                        pick_bbox = np.zeros(4)
                        place_bbox = np.zeros(4)
                elif cmd_id == 1:
                    if action["action"][1] > 150 and is_bbox_inside(pick_bbox, place_bbox, threshold=0.7):
                        # If the gripper is inside the pick or place bbox, we reset the pick bbox
                        print("##########PLACE SUCCESS############", bbox_inside_prcnt(pick_bbox, place_bbox))
                        cmd_index += 1
                        if cmd_index >= total_cmds:
                            break
                        cmd_id = cmd_list[cmd_index]
                        wipe_duration = timestamp  if cmd_id == 4 else wipe_duration
                        goal_pick_label = pick_list[cmd_index]  # "scrub"
                        goal_place_label = place_list[cmd_index]  # "tray"
                        pick_bbox = np.zeros(4)
                        place_bbox = np.zeros(4)                
                elif cmd_id == 2:
                    if action["action"][1] > 150 and action["action"][2] > 140:
                        # If the gripper is inside the pick or place bbox, we reset the pick bbox
                        print("REST SUCCESS")
                        cmd_index += 1
                        if cmd_index >= total_cmds:
                            break
                        cmd_id = cmd_list[cmd_index]
                        wipe_duration = timestamp  if cmd_id == 4 else wipe_duration
                        goal_pick_label = pick_list[cmd_index]  # "scrub"
                        goal_place_label = place_list[cmd_index]  # "tray"
                        pick_bbox = np.zeros(4)
                        place_bbox = np.zeros(4)
                elif cmd_id == 4:
                    if timestamp > wipe_duration + 200:
                        print("WIPING SUCCESS")
                        cmd_index += 1
                        if cmd_index >= total_cmds:
                            break
                        cmd_id = cmd_list[cmd_index]
                        wipe_duration = timestamp  if cmd_id == 4 else wipe_duration
                        goal_pick_label = pick_list[cmd_index]  # "scrub"
                        goal_place_label = place_list[cmd_index]  # "tray"
                        pick_bbox = np.zeros(4)
                        place_bbox = np.zeros(4)

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

                
        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        #log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )

