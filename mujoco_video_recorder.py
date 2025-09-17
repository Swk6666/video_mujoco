"""Utility class to record MuJoCo simulations to MP4 using FFmpeg."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np


class MujocoVideoRecorder:
    """Records MuJoCo frames to a video file using FFmpeg."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fps: float = 60.0,
        resolution: Tuple[int, int] = (2560, 1440),
        output_dir: Path | str = ".",
        filename: str = "simulation.mp4",
        camera_name: str = "fixed_camera",
        encoder: Optional[str] = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")

        self.model = model
        self.data = data
        self.fps = fps

        try:
            width, height = resolution
        except Exception as exc:
            raise ValueError("resolution must be a (width, height) pair") from exc

        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resolution dimensions must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / filename
        self.camera_name = camera_name
        self.encoder = encoder or self._default_encoder()

        self._renderer = mujoco.Renderer(model, self.height, self.width)
        self._scene_option = mujoco.MjvOption()
        self._configure_scene_option()

        self._camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self._camera_id < 0:
            print(f"[MujocoVideoRecorder] Camera '{camera_name}' not found. Using free camera.")

        self._frame_interval = 1.0 / fps
        self._last_frame_time = -np.inf
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        command = self._build_ffmpeg_command()
        try:
            self._ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
            self._active = True
            self._last_frame_time = -np.inf
        except OSError as exc:
            raise RuntimeError("Failed to launch FFmpeg. Ensure ffmpeg is installed and in PATH.") from exc

    def capture_frame(self, force: bool = False) -> bool:
        if not self._active or self._ffmpeg is None or self._ffmpeg.stdin is None:
            raise RuntimeError("Recorder has not been started. Call start() before capture_frame().")

        if not force and self.data.time - self._last_frame_time < self._frame_interval:
            return False

        self._renderer.update_scene(
            self.data,
            camera=self.camera_name if self._camera_id >= 0 else None,
            scene_option=self._scene_option,
        )
        frame = self._renderer.render()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._ffmpeg.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg pipe is broken. Recording cannot continue.") from exc

        self._last_frame_time = self.data.time
        return True

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._ffmpeg is not None:
            stdin = self._ffmpeg.stdin
            if stdin and not stdin.closed:
                stdin.close()
            self._ffmpeg.wait()
        self._active = False
        self._ffmpeg = None

    def __enter__(self) -> "MujocoVideoRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _configure_scene_option(self) -> None:
        flags = self._scene_option.flags
        flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = True

    def _build_ffmpeg_command(self) -> list[str]:
        size = f"{self.width}x{self.height}"
        return [
            "ffmpeg",  # 启动 FFmpeg 可执行文件
            "-y",  # 覆盖已存在的输出文件
            "-f",
            "rawvideo",  # 指定输入流的格式为未压缩原始视频
            "-vcodec",
            "rawvideo",  # 告诉 FFmpeg 输入流使用 rawvideo 编码
            "-s",
            size,  # 输入帧的宽高（例如 2560x1440）
            "-pix_fmt",
            "rgb24",  # 输入帧的像素格式为 8bit RGB
            "-r",
            str(self.fps),  # 输入帧率
            "-i",
            "-",  # 从标准输入读取帧数据
            "-an",  # 不处理音频流
            "-vcodec",
            self.encoder,  # 指定输出视频的编码器
            "-pix_fmt",
            "yuv420p",  # 输出像素格式，兼容大部分播放器
            str(self.output_path),  # 输出文件路径
        ]

    @staticmethod
    def _default_encoder() -> str:
        system = platform.system()
        if system == "Darwin":
            return "h264_videotoolbox"

        if system == "Linux":
            try:
                encoders = subprocess.check_output(["ffmpeg", "-encoders"], text=True)
                if "h264_nvenc" in encoders:
                    return "h264_nvenc"
            except Exception:
                pass
            return "libx264"

        if system == "Windows":
            try:
                encoders = subprocess.check_output(["ffmpeg", "-encoders"], text=True)
                for candidate in ("h264_nvenc", "h264_qsv", "h264_amf"):
                    if candidate in encoders:
                        return candidate
            except Exception:
                pass
            return "libx264"

        return "libx264"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
