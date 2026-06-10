import asyncio
import dataclasses
import logging
import socket
import time
import traceback

import cv2
import numpy as np
import tyro
import websockets.asyncio.server
import websockets.frames

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import msgpack_numpy


CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def camera_aliases(camera_name: str) -> list[str]:
    return [
        camera_name,
        f"rs/{camera_name}",
        f"rs.{camera_name}",
        f"rs_{camera_name}",
    ]


def get_camera_image(images: dict, camera_name: str):
    for alias in camera_aliases(camera_name):
        if alias in images:
            return images[alias]

    rs_images = images.get("rs")
    if isinstance(rs_images, dict) and camera_name in rs_images:
        return rs_images[camera_name]

    raise KeyError(f"Camera '{camera_name}' not found. Available keys: {list(images.keys())}")


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "test").
    config: str
    # Checkpoint directory (e.g., "openpi_checkpoints/test/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the websocket policy server."""

    # Policy loading from a trained checkpoint.
    policy: Checkpoint

    # Fallback prompt used by the policy if the request does not include one.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 55555
    # Record the policy's behavior for debugging.
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 55555,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                receive_start = time.time()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                logging.info("receive time: %.4fs", time.time() - receive_start)

                infer_start = time.time()
                observation = self._build_observation(obs)
                action = self._policy.infer(observation)
                logging.info("infer time: %.4fs", time.time() - infer_start)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _build_observation(self, obs: dict) -> dict:
        observation = {
            "images": {
                camera_name: self.uncompress_image(get_camera_image(obs["images"], camera_name))
                for camera_name in CAMERA_NAMES
            },
            "state": obs["state"],
        }
        if "prompt" in obs:
            observation["prompt"] = obs["prompt"]
        return observation

    def uncompress_image(self, image):
        if isinstance(image, bytes):
            image = np.frombuffer(image, dtype=np.uint8)
        if isinstance(image, np.ndarray) and image.ndim == 1:
            return cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))