# Zerith H1 Pro OpenPI Adapter

This repository adapts the open-source `pi0` model from the [OpenPI](https://github.com/Physical-Intelligence/openpi) project for the Zerith H1 Pro standard robot. It includes the data conversion, training configuration, policy serving, and robot-side inference code used to fine-tune and deploy a vision-language-action policy on Zerith hardware.

The codebase keeps the original OpenPI training stack and adds Zerith-specific pieces for:

- converting Zerith HDF5 demonstrations into LeRobot datasets;
- mapping Zerith multi-camera observations and 23-dimensional joint actions into the `pi0` policy interface;
- fine-tuning the `pi0` LoRA configuration from the public OpenPI base checkpoint;
- serving the trained policy over websocket;
- running closed-loop inference on a Zerith H1 Pro standard robot.

## Hardware and software requirements

### Training machine

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU with CUDA 12 support
- At least 24 GB GPU memory for LoRA fine-tuning; more memory is recommended for larger batch sizes
- `git`, `git-lfs`, and `uv`

### Robot runtime machine

- Zerith H1 Pro standard robot
- Zerith H1 robot SDK and camera SDK installed and importable by the runtime environment
  - The SDK bindings included under `robot_infer/lib/` are based on Zerith SDK `1.3.4`.
  - Robot firmware and SDK APIs may differ between releases. Please choose the SDK version that matches your robot from [zerith_public_sdk](https://github.com/inFpZero/zerith_public_sdk), then replace or rebuild the local bindings as needed.
- Three RGB cameras named:
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`
- Network access to the policy server

The policy server can run on a separate GPU workstation. The robot client sends observations over websocket and receives action chunks from the server.

## Repository layout

```text
.
├── train.sh                         # End-to-end conversion, normalization, and training entry point
├── scripts/
│   ├── convert_new.py               # Zerith HDF5 -> LeRobot conversion
│   ├── compute_norm_stats.py        # Computes state/action normalization stats
│   ├── train.py                     # JAX/OpenPI training loop
│   ├── serve_policy.py              # Original OpenPI websocket policy server
│   └── remote_infer_server.py       # Zerith websocket policy server used for deployment
├── src/openpi/
│   ├── training/config.py           # Zerith training config and dataset settings
│   └── policies/zerith_joint_policy.py
│                                      # Zerith observation/action transforms
├── robot_infer/
│   ├── scripts/test_pi0.py          # Robot-side closed-loop inference client
│   ├── utils/real_env_sdk.py        # Zerith H1 Pro environment wrapper
│   └── lib/                         # Robot communication bindings
├── packages/
│   ├── lerobot/                     # LeRobot dependency used by OpenPI
│   └── openpi-client/               # Websocket client utilities
└── assets/                          # Normalization statistics, grouped by asset_id
```

## Installation

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the Python environment and install the workspace packages:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` avoids downloading large files from the LeRobot submodule during environment setup.

Optional cache locations can be set before training:

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_datasets
export LEROBOT_HOME=/path/to/lerobot_home
export TMPDIR=/path/to/tmp
```

`LEROBOT_HOME` is important because converted datasets are written to:

```text
<LEROBOT_HOME>/<repo_id>
```

### Robot SDK

The repository includes SDK bindings under `robot_infer/lib/` for Zerith SDK `1.3.4`. If your robot runs a different SDK or firmware version, download the matching release from [zerith_public_sdk](https://github.com/inFpZero/zerith_public_sdk) and update the local SDK bindings before running inference.

The robot-side runtime imports these bindings from `robot_infer/lib/`, so make sure the selected SDK can be imported in the robot environment before starting `robot_infer/scripts/test_pi0.py`.

> **SDK 1.3.6+ compatibility note:** The function names in SDK 1.3.6 and above have some subtle changes. If your robot runs SDK 1.3.6+, you may encounter import issues when running on-robot inference. Refer to the [Zerith SDK documentation](https://github.com/inFpZero/zerith_public_sdk) for the exact function names, or load the SDK to inspect available symbols and update the local bindings accordingly.

## Dataset format

The converter expects Zerith demonstrations stored as `.hdf5` files. It recursively scans the directory passed to `--raw_dir` and reads the following fields:

```text
/observation/state/arm/position
/observation/state/effector/position
/observation/state/waist/position
/observation/state/head/position
/observation/state/base/velocity
/observation/images/<camera>/color
```

It also supports camera data under:

```text
/observation/images/rs/<camera>/color
```

The runtime and policy server accept camera keys in either the standard form (`cam_high`) or a RealSense-prefixed form such as `rs/cam_high`, `rs.cam_high`, `rs_cam_high`, or nested under an `rs` dictionary.

## Configuration notes

The main training config is defined in `src/openpi/training/config.py` as:

```python
name="test"
```

The most important path-related fields are:

- `repo_id`: identifies the converted LeRobot dataset. A value such as `zerith/test` maps to `<LEROBOT_HOME>/zerith/test`.
- `assets.assets_dir`: base directory for normalization assets.
- `assets.asset_id`: subdirectory under `assets.assets_dir` where `norm_stats.json` is loaded from. For example, `assets_dir="./assets"` and `asset_id="zerith/test"` maps to `./assets/zerith/test/norm_stats.json`.
- `checkpoint_base_dir`: base directory for checkpoints. With `checkpoint_base_dir="./openpi_checkpoints"`, config name `test`, and experiment name `demo_run`, checkpoints are saved under `./openpi_checkpoints/test/demo_run/`.

> **Tip for `train.sh`:** If you use `train.sh`, make sure `REPO_ID` and `CONFIG_NAME` are consistent with `repo_id` in `src/openpi/training/config.py`. With matching values, the full pipeline (conversion → normalization → training) will run automatically without the "norm stats not found" error.

Keep `repo_id` in `train.sh` aligned with the `repo_id` in the config unless you intentionally train on a different dataset.

## Training

### Option 1: run the full training pipeline

Edit environment variables or pass them inline when running `train.sh`:

```bash
RAW_DIR=/path/to/hdf5_files \
REPO_ID=zerith/my_dataset \
CONFIG_NAME=test \
EXP_NAME=my_first_run \
bash train.sh
```

The script runs three steps:

1. Convert HDF5 demonstrations to a LeRobot dataset.
2. Compute normalization statistics and save them under `assets/<asset_id>/norm_stats.json`.
3. Fine-tune the `pi0` LoRA model and save checkpoints.

Logs are written to `logs/` with timestamps.

### Option 2: run each step manually

Convert the data:

```bash
uv run scripts/convert_new.py \
  --raw_dir /path/to/hdf5_files \
  --repo_id zerith/my_dataset
```

Update `src/openpi/training/config.py` so that `repo_id` and `asset_id` match the dataset you want to train on.

Compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config_name test
```

Start training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py test \
  --exp_name my_first_run \
  --overwrite
```

If you want to overwrite an existing experiment. Otherwise, use `--resume` to continue training from the last checkpoint.

The default config fine-tunes from:

```text
s3://openpi-assets/checkpoints/pi0_base/params
```

By default, only the LoRA parameters are trained and the base model parameters are frozen through the `freeze_filter` in the config.

## Serving a trained policy

After training, choose a checkpoint step directory, for example:

```text
openpi_checkpoints/test/my_first_run/30000
```

Start the Zerith policy server on the GPU machine:

```bash
uv run scripts/remote_infer_server.py policy:checkpoint \
  --policy.config=test \
  --policy.dir=openpi_checkpoints/test/my_first_run/30000 \
  --port=55555
```

The server listens on `0.0.0.0:<port>` and waits for websocket clients. The default port used by `scripts/remote_infer_server.py` and the robot client is `55555`.

Inference is designed to run remotely: the policy server runs on a GPU workstation, while the robot only runs the lightweight websocket client. This setup has been tested on RTX 3090, RTX 4080, RTX 4090, RTX 5080, and newer/higher-end NVIDIA GPUs for policy inference.

`remote_infer_server.py` decodes compressed camera frames sent by the robot client and accepts both standard camera keys (`cam_high`) and RealSense-prefixed keys (`rs/cam_high`, `rs.cam_high`, `rs_cam_high`, or `images["rs"]["cam_high"]`).

## Running inference on Zerith H1 Pro

On the robot runtime machine, make sure the Zerith SDK and camera SDK are installed and that the robot can be initialized by `robot_infer/utils/real_env_sdk.py`.

For deployment, copy only the `robot_infer/` folder to the robot and run it inside the robot's built-in `zerith` conda environment. The full training repository does not need to be installed on the robot.

If the robot environment cannot import `openpi-client`, copy `packages/openpi-client/` from this repository to the robot and install it there:

```bash
cd /path/to/openpi-client
pip install -e .
```

Run the robot-side client:

For best results, use an HDF5 file collected from the same task as the initial pose reference. This helps start the robot from a joint configuration close to the training demonstrations:

```bash
python robot_infer/scripts/test_pi0.py \
  --host <policy-server-ip> \
  --port 55555 \
  --prompt "example" \
  --init_hdf5 /path/to/demo.hdf5 \

```

The client:

1. connects to the policy server;
2. moves the robot to the initial pose;
3. captures the three camera views and current joint state;
4. sends observations to the policy server;
5. smooths action chunks over time;
6. sends joint commands to the Zerith H1 Pro controller at approximately 30 Hz.

Always test with the robot in a safe workspace and keep an emergency stop available.

## Adapting to a new Zerith dataset

For a new task or dataset:

1. Put the raw `.hdf5` demonstrations under a single directory.
2. Choose a unique `repo_id`, for example `zerith/pick_ducks_v1`.
3. Convert the dataset with `scripts/convert_new.py`.
4. Set `repo_id` and `asset_id` in `src/openpi/training/config.py` to the same value.
5. Run `compute_norm_stats.py` before training.
6. Use a new `EXP_NAME` for each training run unless you intentionally pass `--overwrite`.

If your state or action layout changes, update both:

- `scripts/convert_new.py`, where HDF5 fields are assembled into the 23-dimensional state/action vector;
- `src/openpi/policies/zerith_joint_policy.py`, where observations and actions are mapped into the OpenPI policy interface.

## Common issues

### `Norm stats not found`

Run:

```bash
uv run scripts/compute_norm_stats.py --config_name test
```

Check that `assets.asset_id` points to the same directory where the stats were saved.

### Dataset not found

Check `LEROBOT_HOME` and `repo_id`. The dataset should exist at:

```text
<LEROBOT_HOME>/<repo_id>
```

### Policy client cannot connect

Make sure `scripts/remote_infer_server.py` is running, the port is open, and the robot machine can reach the server IP.

### CUDA out of memory

Try reducing `batch_size` in `src/openpi/training/config.py`, lowering the number of visible GPUs, or changing `XLA_PYTHON_CLIENT_MEM_FRACTION`. The default `train.sh` exports `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.

### Action dimension mismatch

The current Zerith adapter uses 23-dimensional state/action vectors. If the robot model or HDF5 layout differs, update the converter and `zerith_joint_policy.py` together.

## Reporting issues

If you run into bugs, SDK compatibility problems, or robot-version-specific issues, please open an issue with the robot model, SDK version, command used, and the relevant logs. Reports are welcome and will be addressed as quickly as possible.

## Acknowledgements

This project builds on the OpenPI codebase and the public `pi0` model released by Physical Intelligence. The Zerith-specific code adapts that training and serving stack for the Zerith H1 Pro standard robot.
