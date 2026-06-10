"""
Convert Zerith H1 Pro HDF5 recordings to the LeRobot dataset format.

Example:
    uv run scripts/convert_new.py --raw_dir /path/to/hdf5_files --repo_id <org>/<dataset-name>
"""

import dataclasses
from datetime import datetime
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro




@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (23,),
            "names": ["states"],
        },
        "action": {
            "dtype": "float32",
            "shape": (23,),
            "names": ["actions"],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observation/images/rs" in ep:
            camera_root = ep["/observation/images/rs"]
        else:
            camera_root = ep["/observation/images"]

        # ignore depth channel, not currently handled
        return [key for key in camera_root.keys() if "depth" not in key]  # noqa: SIM118


def get_camera_color_dataset(ep: h5py.File, camera: str):
    candidates = [
        f"/observation/images/{camera}/color",
        f"/observation/images/rs/{camera}/color",
    ]
    for candidate in candidates:
        if candidate in ep:
            return ep[candidate]

    raise KeyError(
        f"Cannot find image dataset for camera '{camera}'. Tried: {candidates}"
    )


TARGET_IMAGE_SIZE = (640, 480)  # (width, height)


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    target_w, target_h = TARGET_IMAGE_SIZE
    if (w, h) != (target_w, target_h):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}

    for camera in cameras:
        color_dataset = get_camera_color_dataset(ep, camera)
        uncompressed = color_dataset.ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = color_dataset[:]
            # resize per-frame if needed to unify resolution
            if imgs_array.shape[1:3] != (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]):
                imgs_array = np.stack([_resize_if_needed(img) for img in imgs_array])
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in color_dataset:
                img = cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB)
                img = _resize_if_needed(img)
                imgs_array.append(img)
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    str,
]:
    with h5py.File(ep_path, "r") as ep:
        state_arm = torch.from_numpy(ep["/observation/state/arm/position"][:])
        state_effector = torch.from_numpy(ep["/observation/state/effector/position"][:])
        state_waist = torch.from_numpy(ep["/observation/state/waist/position"][:])
        state_head = torch.from_numpy(ep["/observation/state/head/position"][:])
        state_base = torch.from_numpy(ep["/observation/state/base/velocity"][:])
        
        state = torch.cat(
            [
                state_arm[:, :7],
                state_effector[:, :-1],
                state_arm[:, 7:],
                state_effector[:, -1:],
                state_waist[:],
                state_head[:],
                state_base[:],
            ],
            dim=1,
        )
        action = torch.cat([state[1:], state[-1].unsqueeze(0)], dim=0)
        assert action.shape == state.shape, "action.shape should be same with state.shape"

        prompt = ep.attrs.get("task_name", "")

        velocity = None
        effort = None
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort, prompt

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    error_log_path: Path | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    CAMERA_MAPPING = {
        "cam_left_wrist": "cam_left_wrist",
        "cam_right_wrist": "cam_right_wrist",
        "cam_high": "cam_high",
    }
    skipped_episodes: list[tuple[int, Path, str]] = []

    dataset.start_image_writer(num_processes=64, num_threads=8)
    for ep_idx in episodes:
        ep_path = hdf5_files[ep_idx]
        print(f'start episode_{ep_idx}:{ep_path}')

        try:
            imgs_per_cam, state, action, velocity, effort, prompt = load_raw_episode_data(ep_path)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            skipped_episodes.append((ep_idx, ep_path, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] Skip episode_{ep_idx} due to read/process error: {ep_path}")
            print(f"[WARN] {type(exc).__name__}: {exc}")
            continue

        num_frames = state.shape[0]

        for i in tqdm.tqdm(range(num_frames)):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():

                target_camera = CAMERA_MAPPING.get(camera, camera)
                frame[f"observation.images.{target_camera}"] = img_array[i]
            dataset.add_frame(frame)

        dataset.save_episode(task=prompt)

    dataset.stop_image_writer()

    if skipped_episodes:
        if error_log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log_path = Path("logs") / f"convert_skipped_{timestamp}.log"

        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with error_log_path.open("w", encoding="utf-8") as f:
            f.write("Skipped files during conversion\n")
            for ep_idx, ep_path, message in skipped_episodes:
                f.write(f"episode_{ep_idx}\t{ep_path}\t{message}\n")

        print(f"[WARN] Skipped {len(skipped_episodes)} episodes. Error log: {error_log_path}")

    return dataset


def find_hdf5_in_leaves(root_dir):
    """Collect all .hdf5 files recursively under root_dir."""
    root_path = Path(root_dir)
    hdf5_files = sorted(root_path.rglob("*.hdf5"))
    print(f"[INFO] Found {len(hdf5_files)} hdf5 files under {root_path}")
    return hdf5_files


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "serve tea",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    error_log_path: Path | None = None,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)
    
    hdf5_files = find_hdf5_in_leaves(raw_dir)
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found under {raw_dir}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_arx",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        error_log_path=error_log_path,
    )
    dataset.consolidate(run_compute_stats=False)


if __name__ == "__main__":
    tyro.cli(port_aloha)