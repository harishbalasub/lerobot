#!/usr/bin/env python

import json
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import logging
import os

def update_json_metadata(dataset_root, repo_id):
    """Update JSON metadata with observation.state and next_observation.state from parquet files."""
    # Initialize dataset
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    logging.info(f"Loaded dataset with {dataset.num_episodes} episodes")

    # Ensure meta directory exists
    meta_dir = Path(dataset_root) / "meta" / "pickplace_bbox"
    meta_dir.mkdir(parents=True, exist_ok=True)


    num_episodes = len(os.listdir(meta_dir))

    # Iterate over episodes
    #for ep_idx in range(num_episodes):
    for ep_idx in range(195, num_episodes):
        # Filter dataset for episode


        ep_data = [item for item in dataset.hf_dataset if item["episode_index"].item() == ep_idx]
        if not ep_data:
            logging.warning(f"No data found for episode {ep_idx}")
            continue

        # Load existing JSON metadata
        json_path = meta_dir / f"episode_{ep_idx:06d}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = []
            logging.warning(f"No existing metadata for episode {ep_idx}; creating new")

        # Ensure metadata length matches episode data
        if len(metadata) != len(ep_data):
            print(f"json file name : {json_path}")

            logging.warning(f"Mismatch: episode {ep_idx} has {len(ep_data)} frames, metadata has {len(metadata)}")
            metadata = [metadata[i] for i in range(min(len(metadata),len(ep_data)))]

        # Update metadata with observation.state and next_observation.state
        for frame_idx, (item, meta_item) in enumerate(zip(ep_data, metadata)):
            # Current observation.state
            meta_item["observation.state"] = item["observation.state"].tolist()
            # Next observation.state
            next_frame_idx = frame_idx + 1
            if next_frame_idx < len(ep_data)-1:
                meta_item["next_observation.state"] = ep_data[next_frame_idx]["observation.state"].tolist()
            else:
                meta_item["next_observation.state"] = item["observation.state"].tolist()  # Terminal state

        # Save updated metadata
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Updated metadata for episode {ep_idx}")

        #input("Press Enter to continue to the next episode...")

if __name__ == "__main__":
    dataset_root = "/home/harishbalasub/.cache/huggingface/lerobot/harishbalasub/so100_diverse_tasks"  # Update with your dataset path
    repo_id = "harishbalasub/so100_diverse_tasks"  # Update with your repo ID
    update_json_metadata(dataset_root, repo_id)