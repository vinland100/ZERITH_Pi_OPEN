#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Unified WebRTC + gRPC client (RealSense + V4L2)
- Synchronous API unchanged.
- Internally more stable: fewer copies, timeouts, clean shutdown, gRPC keepalive.
"""

from __future__ import annotations
import asyncio
import logging
import threading
import time
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque

import av
import numpy as np

# gRPC
import grpc
import grpc.aio as grpc_aio

# WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

# Unified proto generated from robot_service.proto
from . import robot_pb2 as pb
from . import robot_pb2_grpc as rpc

logger = logging.getLogger("unified_client")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="[unified_client] %(asctime)s %(levelname)s: %(message)s")


# ============================================================
#                 Internal: async WebRTC receiver
# ============================================================

class _VideoReceiverAsync:
    """
    Internal use:
      - Two-step WebRTC handshake (server as Offerer).
      - Async video reception → write to thread-safe single-frame buffer (readable from sync side).
    """
    def __init__(self, aio_stub: rpc.RobotServiceStub,
                 frame_bufs: Dict[str, deque],
                 buf_lock: threading.Lock):
        self._stub = aio_stub
        self._pc: Optional[RTCPeerConnection] = None
        self._video_tracks: Dict[str, object] = {}
        self._track_tasks: Dict[str, asyncio.Task] = {}
        self._closed = asyncio.Event()

        # Buffer shared with the external side (one frame per camera).
        self._frame_bufs = frame_bufs
        self._buf_lock = buf_lock

    @property
    def pc(self) -> Optional[RTCPeerConnection]:
        return self._pc

    async def connect(self):
        cfg = RTCConfiguration(iceServers=[])
        pc = RTCPeerConnection(configuration=cfg)
        self._pc = pc

        @pc.on("connectionstatechange")
        async def _on_state():
            logger.info("[PC] state → %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close()

        @pc.on("track")
        def _on_track(track):
            cam_name = getattr(track, "id", "<unknown>")
            logger.info("[PC] ontrack: kind=%s id=%s", track.kind, cam_name)
            if track.kind == "video":
                self._video_tracks[cam_name] = track
                task = asyncio.create_task(self._consume_video(track, cam_name))
                self._track_tasks[cam_name] = task

                @track.on("ended")
                async def _on_ended():
                    logger.info("[PC] track ended: %s", cam_name)
                    self._video_tracks.pop(cam_name, None)
                    t = self._track_tasks.pop(cam_name, None)
                    if t:
                        t.cancel()

        # Step-1: request offer.
        req1 = pb.ControlSignalRequest(action=pb.ControlSignalRequest.CONNECT, sdp_type="", sdp="")
        resp1 = await self._stub.ControlVideo(req1, timeout=10)
        if resp1.status != "ok" or not resp1.sdp or resp1.sdp_type != "offer":
            await self.close()
            raise RuntimeError(f"[GRPC] invalid offer: status={resp1.status}, type={resp1.sdp_type}")

        await pc.setRemoteDescription(RTCSessionDescription(sdp=resp1.sdp, type=resp1.sdp_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Step-2: send back answer.
        req2 = pb.ControlSignalRequest(
            action=pb.ControlSignalRequest.CONNECT,
            sdp_type=pc.localDescription.type,
            sdp=pc.localDescription.sdp,
        )
        resp2 = await self._stub.ControlVideo(req2, timeout=10)
        if resp2.status != "ok":
            await self.close()
            raise RuntimeError(f"[GRPC] answer rejected: status={resp2.status}")

        logger.info("[PC] connected. waiting for video tracks...")

    async def _consume_video(self, track, cam_name: str):
        """
        Safer consumption:
        - recv() with 1s timeout, periodically checks self._closed.
        - Exits cleanly on cancellation or exception.
        - Single-frame ring buffer (no copy on write, copy on read).
        """
        try:
            while not self._closed.is_set():
                try:
                    frame: av.VideoFrame = await asyncio.wait_for(track.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("[PC] recv error on %s: %s", cam_name, e)
                    break

                try:
                    # to_ndarray already returns new memory; no .copy() to reduce write overhead.
                    bgr: np.ndarray = frame.to_ndarray(format="bgr24")
                except Exception as e:
                    logger.warning("[PC] to_ndarray failed on %s: %s", cam_name, e)
                    continue

                ts = time.monotonic()
                with self._buf_lock:
                    if cam_name not in self._frame_bufs:
                        self._frame_bufs[cam_name] = deque(maxlen=1)
                    dq = self._frame_bufs[cam_name]
                    dq.clear()
                    dq.append((bgr, ts))  # No copy on write.
        finally:
            try:
                track.stop()
            except Exception:
                pass

    async def disconnect(self):
        try:
            req = pb.ControlSignalRequest(action=pb.ControlSignalRequest.DISCONNECT, sdp_type="", sdp="")
            await self._stub.ControlVideo(req, timeout=5)
        except Exception as e:
            logger.info("[GRPC] DISCONNECT error (ignored): %s", e)

    async def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        for t in list(self._track_tasks.values()):
            t.cancel()
        self._track_tasks.clear()
        if self._pc:
            await self._pc.close()
            self._pc = None
        logger.info("[PC] closed.")


# ============================================================
#                 External: synchronous unified client
# ============================================================

class UnifiedReceiverClient:
    """
    Unified client (RealSense + V4L2)
    API:
      - start()/stop()
      - get_latest_frame(cam_name) -> (bgr8, ts) or None         # cam_name is the plain name (e.g. "cam_high")
      - get_state(camera_names=None) -> RecorderGetStateReply
      - reinit_rs()/reinit_rs_and_wait()
      - reinit_v4l2()/reinit_v4l2_and_wait()
    """
    def __init__(self,
                 grpc_target: str = "localhost:50051",
                 connect_timeout: float = 10.0):
        self._grpc_target = grpc_target
        self._connect_timeout = float(connect_timeout)

        # ---- Sync gRPC (blocking) — used for get_state / reinit ----
        self._channel_sync = grpc.insecure_channel(
            self._grpc_target,
            options=[
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.max_receive_message_length", 40 * 1024 * 1024),
                ("grpc.max_send_message_length", 40 * 1024 * 1024),
            ],
        )
        self._stub_sync = rpc.RobotServiceStub(self._channel_sync)

        # ---- Async gRPC (ControlVideo) created in the event loop thread ----
        self._aio_channel: Optional[grpc_aio.Channel] = None
        self._aio_stub: Optional[rpc.RobotServiceStub] = None

        # ---- Single-frame ring buffer per camera (writer in async coroutine, reader in main thread) ----
        # Before:
        # self._frame_bufs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1))

        # After:
        self._frame_bufs: Dict[str, deque] = {}

        self._buf_lock = threading.Lock()

        # ---- Event loop thread & control ----
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_req = threading.Event()
        self._started_evt = threading.Event()
        self._start_exc: Optional[BaseException] = None

        self._receiver_async: Optional[_VideoReceiverAsync] = None

    # ---------- Sync: start / stop ----------
    def start(self) -> None:
        """
        Start the background event loop thread and establish the WebRTC connection
        (blocks until completion or error).
        """
        if self._thread and self._thread.is_alive():
            return

        self._stop_req.clear()
        self._started_evt.clear()
        self._start_exc = None

        def _thread_main():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._async_bootstrap())
            except BaseException as e:
                self._start_exc = e
                self._started_evt.set()
                logger.exception("start thread bootstrap error")
                return

            # Connection complete, notify the main thread.
            self._started_evt.set()

            # Main loop: wait for stop.
            try:
                self._loop.run_until_complete(self._async_wait_stop())
            finally:
                try:
                    self._loop.run_until_complete(self._async_shutdown())
                finally:
                    self._loop.stop()
                    self._loop.close()

        self._thread = threading.Thread(target=_thread_main, name="Unified-Receiver", daemon=True)
        self._thread.start()

        # Wait for connection result.
        ok = self._started_evt.wait(timeout=self._connect_timeout + 5.0)
        if not ok:
            raise TimeoutError("start() timed out waiting for WebRTC connect")
        if self._start_exc:
            raise RuntimeError(f"start() failed: {self._start_exc}") from self._start_exc

    def stop(self) -> None:
        """
        Stop the background thread and clean up resources (synchronous, blocks until finished).
        """
        if not self._thread:
            return
        self._stop_req.set()
        # Make the event loop wake up quickly.
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass
        self._thread.join(timeout=self._connect_timeout + 5.0)
        self._thread = None

    # ---------- Sync: image frame ----------
    def get_latest_frame(self, cam_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest (bgr8, timestamp) for cam_name.
        Note: cam_name is a plain name without "rs/" or "v4l2/" prefix (aligned with track.id).
        """
        with self._buf_lock:
            dq = self._frame_bufs.get(cam_name)
            if not dq or not dq:
                return None
            # Copy on read to prevent external modification.
            bgr, ts = dq[-1]
            return bgr.copy(), ts

    # ---------- Sync: state and reconfiguration ----------
    def get_state(self,
                  camera_names: Optional[List[str]] = None,
                  
                  timeout: float = 5.0) -> pb.RecorderGetStateReply:
        req = pb.RecorderGetStateRequest(camera_names=(camera_names or []))
        return self._stub_sync.RecorderGetState(req, timeout=timeout)

    # --- RealSense ---
    def reinit_rs(self, targets: List[Dict], timeout: float = 5.0) -> pb.RecorderReinitReply:
        req = pb.RecorderReinitRequest(targets=[
            pb.RecorderReinitTarget(
                rs=pb.RsRecorderTarget(
                    camera_name=str(t["camera_name"]),
                    streams=[
                        pb.RsStreamSpec(
                            type=str(s["type"]),
                            width=int(s["width"]),
                            height=int(s["height"]),
                            fps=int(s["fps"]),
                            fmt=str(s.get("fmt", "any")).lower(),
                        )
                        for s in (t.get("streams") or [])
                    ]
                )
            ) for t in (targets or [])
        ])
        return self._stub_sync.RecorderReinit(req, timeout=timeout)

    def reinit_rs_and_wait(self,
                           targets: List[Dict],
                           wait_timeout: float = 6.0,
                           poll_interval: float = 0.2) -> Tuple[pb.RecorderReinitReply, bool]:
        reply = self.reinit_rs(targets)
        if not getattr(reply, "ok", False):
            return reply, False

        # Expected: cam -> {type -> (w,h,fps,fmt_lower)}
        expected: Dict[str, Dict[str, Tuple[int,int,int,str]]] = {}
        for t in (targets or []):
            nm = str(t["camera_name"])
            mm: Dict[str, Tuple[int,int,int,str]] = {}
            for s in (t.get("streams") or []):
                st = str(s["type"]).lower()
                mm[st] = (int(s["width"]), int(s["height"]), int(s["fps"]),
                          str(s.get("fmt", "any")).lower())
            expected[nm] = mm

        deadline = time.time() + float(wait_timeout)
        while time.time() < deadline:
            try:
                names = list(expected.keys())
                st = self.get_state(names, timeout=wait_timeout)
                # Actual: cam -> {type -> (w,h,fps,fmt_lower)}
                actual_map: Dict[str, Dict[str, Tuple[int,int,int,str]]] = {}
                for cfg in st.actuals:
                    if cfg.WhichOneof("detail") == "rs":
                        amap: Dict[str, Tuple[int,int,int,str]] = {}
                        for sp in cfg.rs.streams:
                            amap[sp.type.lower()] = (int(sp.width), int(sp.height), int(sp.fps),
                                                     str(sp.fmt or "").lower())
                        actual_map[cfg.rs.camera_name] = amap
                # Check subset match.
                all_ok = True
                for cam, emap in expected.items():
                    amap = actual_map.get(cam, {})
                    ok_cam = True
                    for stype, e in emap.items():
                        a = amap.get(stype)
                        if a != e:
                            ok_cam = False
                            break
                    if not ok_cam:
                        all_ok = False
                        break

                if all_ok:
                    return reply, True

            except Exception as e:
                logger.warning("[reinit_rs_and_wait] get_state error: %s", e)

            time.sleep(float(poll_interval))

        return reply, False

    # --- V4L2 ---
    def reinit_v4l2(self, targets: List[Dict], timeout: float = 5.0) -> pb.RecorderReinitReply:
        req = pb.RecorderReinitRequest(targets=[
            pb.RecorderReinitTarget(
                v4l2=pb.V4L2RecorderTarget(
                    camera_name=str(t["camera_name"]),
                    stream=pb.V4L2StreamSpec(
                        width=int(t.get("width", 0)),
                        height=int(t.get("height", 0)),
                        fps=int(t.get("fps", 0)),
                        pixel_format_fourcc=str(t.get("fourcc", "")),
                    )
                )
            ) for t in (targets or [])
        ])
        return self._stub_sync.RecorderReinit(req, timeout=timeout)

    def reinit_v4l2_and_wait(self,
                             targets: List[Dict],
                             wait_timeout: float = 6.0,
                             poll_interval: float = 0.2) -> Tuple[pb.RecorderReinitReply, bool]:
        reply = self.reinit_v4l2(targets)
        if not getattr(reply, "ok", False):
            return reply, False

        # Expected: cam -> (w,h,fps,FOURCC_UPPER)
        expected: Dict[str, Tuple[int,int,int,str]] = {}
        for t in (targets or []):
            nm = str(t["camera_name"])
            w  = int(t.get("width", 0))
            h  = int(t.get("height", 0))
            f  = int(t.get("fps", 0))
            cc = str(t.get("fourcc", "")).upper()
            expected[nm] = (w, h, f, cc)

        deadline = time.time() + float(wait_timeout)
        while time.time() < deadline:
            try:
                names = list(expected.keys())
                st = self.get_state(names, timeout=wait_timeout)
                # Actual: cam -> (w,h,fps,FOURCC_UPPER)
                actual_map: Dict[str, Tuple[int,int,int,str]] = {}
                for cfg in st.actuals:
                    if cfg.WhichOneof("detail") == "v4l2":
                        s = cfg.v4l2.stream
                        actual_map[cfg.v4l2.camera_name] = (
                            int(s.width), int(s.height), int(s.fps),
                            str(s.pixel_format_fourcc or "").upper()
                        )
                # Check full match.
                all_ok = True
                for cam, want in expected.items():
                    got = actual_map.get(cam)
                    if got != want:
                        all_ok = False
                        break
                if all_ok:
                    return reply, True

            except Exception as e:
                logger.warning("[reinit_v4l2_and_wait] get_state error: %s", e)

            time.sleep(float(poll_interval))

        return reply, False

    # ---------- Coroutines running inside the event loop thread ----------
    async def _async_bootstrap(self):
        # Async gRPC channel & stub (only used for ControlVideo).
        self._aio_channel = grpc_aio.insecure_channel(
            self._grpc_target,
            options=(
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.max_receive_message_length", 40 * 1024 * 1024),
                ("grpc.max_send_message_length", 40 * 1024 * 1024),
            ),
        )
        self._aio_stub = rpc.RobotServiceStub(self._aio_channel)

        # Set up WebRTC receiver and connect.
        # Before:
        self._receiver_async = _VideoReceiverAsync(
            aio_stub=self._aio_stub,
            frame_bufs=self._frame_bufs,
            buf_lock=self._buf_lock,
        )

        # # After:
        # # Convert defaultdict to plain dict.
        # frame_bufs_dict = dict(self._frame_bufs)
        # self._receiver_async = _VideoReceiverAsync(
        #     aio_stub=self._aio_stub,
        #     # frame_bufs=frame_bufs_dict,
        #     frame_bufs=self._frame_bufs,  # Pass reference directly.
        #     buf_lock=self._buf_lock,
        # )
        await self._receiver_async.connect()

    async def _async_wait_stop(self):
        # Poll for stop request (short sleep to stay responsive).
        while not self._stop_req.is_set():
            await asyncio.sleep(0.1)

    async def _async_shutdown(self):
        try:
            if self._receiver_async:
                await self._receiver_async.disconnect()
                await self._receiver_async.close()
        finally:
            if self._aio_channel:
                await self._aio_channel.close()
