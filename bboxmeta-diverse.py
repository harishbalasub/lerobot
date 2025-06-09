import json
from pathlib import Path
import cv2
from ultralytics import YOLO
import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CONFIG — SET THESE
dataset_root = Path("/home/harishbalasub/.cache/huggingface/lerobot/harishbalasub/so100_diverse_tasks")  # 👈 CHANGE THIS

gripper_label = "gripper"

# Paths
video_dir = dataset_root / "videos" / "chunk-000" / "observation.images.laptop"
meta_dir = dataset_root / "meta" / "pickplace_bbox"
meta_dir.mkdir(parents=True, exist_ok=True)

frames_dir = "frames"
# Load YOLOv8 model
model = YOLO("yolotrain/runs/detect/train6/weights/best.pt")

def read_cmd_info():
    meta_cmd_info_path = dataset_root / "meta" / "episodes.jsonl"
    cmd_list = []
    pick_label_list = []
    place_label_list = []
    num_episodes = 0
    if meta_cmd_info_path.exists():
        with open(meta_cmd_info_path, "r") as f:
            for line in f:
                
                # each line is like following
                # {"episode_index": 0, "tasks": ["cmd 1: Place scrub in bowl"], "length": 579}
                # i want to get tasks[0] string
                task_string = json.loads(line.strip())["tasks"][0]
                # parse the task string to get the pick and place labels
                parts = task_string.split(" ")
                if len(parts) < 2 or parts[0] != "cmd":
                    assert False, f"Invalid task string: {task_string}"
                num_episodes += 1
                cmd_id = int(parts[1].split(":")[0].strip())
                if cmd_id == 0 or cmd_id == 3:
                    pick_label = parts[3].strip()
                    place_label = ""
                elif cmd_id == 1:
                    pick_label = parts[3].strip()
                    place_label = parts[5].strip()
                elif cmd_id == 2 or cmd_id == 4:
                    pick_label = ""
                    place_label = ""
                else:
                    assert False, f"Invalid command ID: {cmd_id} in task string: {task_string}"

                cmd_list.append(cmd_id)
                pick_label_list.append(pick_label)
                place_label_list.append(place_label)
                print(f"Episode {num_episodes}: cmd {cmd_id}, pick {pick_label}, place {place_label}")

    # return as dict of 'num_episodes', 'cmd_list', 'pick_label_list', 'place_label_list'
    meta_cmd_dict = {
        "num_episodes": num_episodes,
        "cmd_list": cmd_list,
        "pick_label_list": pick_label_list,
        "place_label_list": place_label_list
    }
    return meta_cmd_dict

def extract_bbox(boxes, names, label, prev_bbox):
    for box in boxes:
        if names[int(box.cls)] == label:
            return list(map(float, box.xyxy[0].tolist()))
    return prev_bbox

def is_inside(a, b, threshold=0.8):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    prcnt = inter / area_a >= threshold
    #print(f"is_inside: a {a}, b {b}, prcnt {inter/area_a}")
    return prcnt

def compute_reward(pick, place):
    return float(is_inside(pick, place))

def check_pick_success(pick_class, place_class, boxes, names, threshold=0.3):
    return check_place_success(pick_class, place_class, boxes, names, threshold=threshold)

def check_place_success(pick_class, place_class, boxes, names, threshold=0.8):
    # Parse bounding boxes
    pick_boxes = []
    place_boxes = []
    for box in boxes:
        cls_id = int(box.cls)
        cls_name = names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if cls_name == pick_class:
            pick_boxes.append([x1, y1, x2, y2])
        elif cls_name == place_class:
            place_boxes.append([x1, y1, x2, y2])

    # Evaluate condition: pick is 80% inside any place box
    for pick_box in pick_boxes:
        for place_box in place_boxes:
            if is_inside(pick_box, place_box, threshold=threshold):
                return True
    return False

for frame in os.listdir(frames_dir):
    os.remove(os.path.join(frames_dir, frame))


meta_cmd_info = read_cmd_info()
if meta_cmd_info["num_episodes"] != len(os.listdir(video_dir)):
    print(f"Warning: Number of episodes in metadata ({meta_cmd_info['num_episodes']}) does not match number of video files ({len(os.listdir(video_dir))}).")
#input("Press Enter to continue...")
# Main loop: process all episodes
# for each video file in video_dir files
#for video_path in video_dir.glob("*.mp4"): 
#video_select = [11,  15, 16, 20, 21, 25, 26, 27, 52, 56, 57, 64, 66, 67, 68, 71, 72, 79, 83]
#video_select = [52, 56, 57, 64, 66, 67, 68, 71, 72, 79, 83]
#video_select = [2]
#video_select = [42, 45, 59, 62, 73]
#video_select = [96, 104, 199]
#for vidx in range(len(os.listdir(video_dir))):
for vidx in range(195, len(os.listdir(video_dir))):    
#for vidx in video_select:

 
    video_path = video_dir / f"episode_{vidx:06d}.mp4"

    # use ffmpeg system command to save the frames of video_path to frames_dir
    os.system(f"ffmpeg -i {video_path} -pix_fmt rgb24 frames/frame_%04d.png -loglevel quiet")

    
    
    # get the basename of the video file
    episode_name = f"episode_{vidx:06d}"
    episode_data = []
    frame_idx = 0
    reward_success = 0
    rewards_per_episode = 1 #10
    pick_rewards_per_episode = 1
    reward_success_flag = False    
    done = 0
    goal_pick_label = meta_cmd_info["pick_label_list"][vidx]
    goal_place_label = meta_cmd_info["place_label_list"][vidx]    
    pick = [0, 0, 0, 0]
    place = [0, 0, 0, 0]
    gripper = [0, 0, 0, 0]
    cmd_id = meta_cmd_info["cmd_list"][vidx]
    num_frames = len(os.listdir(frames_dir))
    print(f"Num frames in {video_path}: {num_frames}")
    print(f"cmd id: {cmd_id}, goal pick label: {goal_pick_label}, goal place label: {goal_place_label}")
    for idx in range(num_frames):
        
        frame = f"frame_{idx+1:04d}.png"
        #print(f"Processing {frame}")
        if not os.path.exists(os.path.join(frames_dir, frame)):
            print(f"Frame {frame} does not exist, skipping...")
            continue
        # read the image

        frame_img = cv2.imread(os.path.join(frames_dir, frame))
        # change the color space from BGR to RGB
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # resize the image to 640x640
        #frame_img = cv2.resize(frame_img, (640, 640))
        #model = YOLO("yolotrain/runs/detect/train/weights/best.pt")
        results = model.predict(frame_img, imgsz=640, conf=0.25, verbose=False)[0]
        boxes = results.boxes
        names = model.names

        
        annotated_img = results.plot()  # Visualize YOLO detections
        

        
        '''        
        if idx % 10 == 0:
            plt.imshow(annotated_img)
            plt.axis('off')
            plt.show(block=True)
            plt.savefig(f"yolo_output_{episode_name}_{idx}.png")
            input("Press Enter to continue...")
        '''
        
        #print(f"frame {frame_idx}: pick {goal_pick_label} place {goal_place_label} {len(boxes)} boxes detected.")
        if cmd_id == 0 or cmd_id == 3:
            pick = extract_bbox(boxes, names, goal_pick_label, pick)
            place = [0, 0, 0, 0]
            gripper = extract_bbox(boxes, names, gripper_label, gripper)
            if not reward_success_flag and (idx > 2*num_frames/3) and check_pick_success(gripper_label, goal_pick_label, boxes, names):            
                reward = 0.0
                reward_success += 1

                if reward_success >= pick_rewards_per_episode:
                    print(f"Episode {episode_name} cmd {cmd_id} completed successfully.")
                    reward_success_flag = True                
                    reward = 2.0
                    #plt.imshow(annotated_img)
                    #plt.axis('off')
                    #plt.show(block=False)
                    #input("Episode success")
            else:
                reward = -0.01
                reward_success = 0
                # if gripper is not empty and gripper is not overlapping with pick, then reward is -0.1
                if gripper != [0, 0, 0, 0] and pick != [0, 0, 0, 0] and not (is_inside(gripper, pick, threshold=0.3) or is_inside(pick, gripper, threshold=0.3)):
                    reward += -0.1
                #if gripper != [0, 0, 0, 0] and place != [0, 0, 0, 0]  and not (is_inside(gripper, place, threshold=0.5) or is_inside(pick, place, threshold=0.5)):
                #    reward += -0.1

            if idx == len(os.listdir(frames_dir)) - 1:
                done = 1                
                if not reward_success_flag:
                    print(f"Episode {episode_name} failed.")
                    #plt.imshow(annotated_img)
                    #plt.axis('off')
                    #plt.show(block=True)
                    pass
                else:
                    reward = 1.0
        elif cmd_id == 1:
            pick = extract_bbox(boxes, names, goal_pick_label, pick)
            place = extract_bbox(boxes, names, goal_place_label, place)
            gripper = extract_bbox(boxes, names, gripper_label, gripper)
            #reward = compute_reward(pick, place)
            if not reward_success_flag and (idx > 2*num_frames/3) and check_place_success(goal_pick_label, goal_place_label, boxes, names):            
                reward = 0.0
                reward_success += 1

                if reward_success >= rewards_per_episode:
                    print(f"Episode {episode_name} cmd {cmd_id} completed successfully.")
                    reward_success_flag = True                
                    reward = 2.0
                    #plt.imshow(annotated_img)
                    #plt.axis('off')
                    #plt.show(block=False)
                    #input("Episode success")
            else:
                reward = -0.01
                reward_success = 0
                # if gripper is not empty and gripper is not overlapping with pick, then reward is -0.1
                #if gripper != [0, 0, 0, 0] and pick != [0, 0, 0, 0] and not (is_inside(gripper, pick, threshold=0.3) or is_inside(pick, gripper, threshold=0.3)):
                #    reward += -0.1
                #if gripper != [0, 0, 0, 0] and place != [0, 0, 0, 0]  and not (is_inside(gripper, place, threshold=0.5) or is_inside(pick, place, threshold=0.5)):
                #    reward += -0.1

            if idx == len(os.listdir(frames_dir)) - 1:
                done = 1                
                if not reward_success_flag:
                    print(f"Episode {episode_name} failed.")
                    pass
                else:
                    reward = 1.0
                    
        elif cmd_id == 2:
            pick = [0, 0, 0, 0]
            place = [0, 0, 0, 0]
            gripper = extract_bbox(boxes, names, gripper_label, gripper)
            reward = -.01 
            if idx == len(os.listdir(frames_dir)) - 1:
                done = 1
                reward = 1.0
                reward_success_flag = True
                print(f"Episode {episode_name} cmd {cmd_id} completed successfully.")
        elif cmd_id == 4:
            pick = [0, 0, 0, 0]
            place = [0, 0, 0, 0]
            gripper = extract_bbox(boxes, names, gripper_label, gripper)
            reward = -.01 
            if idx == len(os.listdir(frames_dir)) - 1:
                done = 1
                reward = 1.0
                reward_success_flag = True
                print(f"Episode {episode_name} cmd {cmd_id} completed successfully.")

        else:
            assert False, f"Invalid command ID: {cmd_id} in episode {episode_name}. Only cmd 1 is supported for pick and place tasks."

        episode_data.append({
            "cmd": cmd_id,
            "goal.pick_bbox": pick,
            "goal.place_bbox": place,
            "gripper_bbox": gripper,
            "reward": reward,
            "done": done,
        })

        #print(f"Frame {frame}: Pick: {pick}, Place: {place}, Gripper: {gripper}, Reward: {reward}")
        frame_idx += 1
        # remove the frame file
        #input("Press Enter to continue to next frame...")
        
        

    
    # remove all files in frames_dir
    for frame in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame))
    out_path = meta_dir / f"{episode_name}.json"
    #print(f"Number of frames in {episode_name}: {len(episode_data)}")
    #input("Wait before dumping metadata...")
    with open(out_path, "w") as f:
        json.dump(episode_data, f, indent=2)
    #input("Press Enter to continue...")

print("✅ Finished generating YOLO bbox + reward metadata.")