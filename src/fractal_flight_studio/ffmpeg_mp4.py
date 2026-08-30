from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable, Iterable, Protocol
from uuid import uuid4

import mpmath as mp
import numpy as np


ProgressCallback = Callable[["Mp4ExportProgress"], None]
CancellationCheck = Callable[[], bool]


class _RgbFrame(Protocol):
    index: int
    time_seconds_text: str
    rgb: np.ndarray


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegProbeError(RuntimeError):
    pass


class Mp4ExportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        return_code: int | None = None,
        stderr: str = "",
    ) -> None:
        self.return_code = return_code
        self.stderr = stderr
        detail = f"\nFFmpeg: {stderr.strip()}" if stderr.strip() else ""
        super().__init__(message + detail)


class Mp4ExportCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FFmpegInfo:
    executable: str
    version_line: str


@dataclass(frozen=True, slots=True)
class Mp4ExportSettings:
    ffmpeg_executable: str = "ffmpeg"
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    output_pixel_format: str = "yuv420p"
    faststart: bool = True
    overwrite: bool = False
    extra_output_args: tuple[str, ...] = ()
    shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.ffmpeg_executable.strip():
            raise ValueError("FFmpeg executable must not be empty")
        if not self.video_codec.strip():
            raise ValueError("video codec must not be empty")
        if not self.preset.strip():
            raise ValueError("video preset must not be empty")
        if not 0 <= self.crf <= 51:
            raise ValueError("video CRF must be between 0 and 51")
        if not self.output_pixel_format.strip():
            raise ValueError("output pixel format must not be empty")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown timeout must be positive")
        if any(not argument for argument in self.extra_output_args):
            raise ValueError("extra FFmpeg arguments must not be empty")


@dataclass(frozen=True, slots=True)
class Mp4ExportPlan:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    duration_text: str
    digits: int
    frame_count: int
    source_frame_count: int

    def time_seconds_text(self, index: int) -> str:
        if not 0 <= index < self.frame_count:
            raise IndexError("MP4 frame index out of range")
        value = Fraction(index * self.fps_denominator, self.fps_numerator)
        with mp.workdps(self.digits):
            numeric = mp.mpf(value.numerator) / value.denominator
            return mp.nstr(numeric, n=self.digits, min_fixed=-6, max_fixed=6)


@dataclass(frozen=True, slots=True)
class Mp4ExportProgress:
    frames_written: int
    total_frames: int
    last_frame_index: int
    time_seconds_text: str
    bytes_written: int


@dataclass(frozen=True, slots=True)
class Mp4ExportResult:
    output_path: Path
    frame_count: int
    bytes_written: int
    elapsed_seconds: float
    ffmpeg: FFmpegInfo
    command: tuple[str, ...]


def resolve_ffmpeg(executable: str = "ffmpeg") -> str:
    candidate = Path(executable).expanduser()
    contains_separator = any(
        separator and separator in executable for separator in (os.sep, os.altsep)
    )
    if candidate.is_absolute() or contains_separator:
        if not candidate.is_file():
            raise FFmpegNotFoundError(f"FFmpeg executable not found: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(executable)
    if resolved is None:
        raise FFmpegNotFoundError(
            f"FFmpeg executable '{executable}' was not found on PATH"
        )
    return resolved


def probe_ffmpeg(
    executable: str = "ffmpeg", *, timeout_seconds: float = 10.0
) -> FFmpegInfo:
    resolved = resolve_ffmpeg(executable)
    try:
        completed = subprocess.run(
            [resolved, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegProbeError(f"could not run FFmpeg version probe: {exc}") from exc
    output = completed.stdout or ""
    version_line = output.splitlines()[0].strip() if output.splitlines() else ""
    if completed.returncode != 0 or not version_line:
        raise FFmpegProbeError(
            "FFmpeg version probe failed with exit code "
            f"{completed.returncode}: {output.strip()}"
        )
    return FFmpegInfo(resolved, version_line)


def build_ffmpeg_mp4_command(
    ffmpeg: FFmpegInfo,
    plan: Mp4ExportPlan,
    temporary_output: Path,
    settings: Mp4ExportSettings,
) -> tuple[str, ...]:
    if settings.output_pixel_format in {"yuv420p", "yuv420p10le"} and (
        plan.width % 2 or plan.height % 2
    ):
        raise ValueError(
            f"{settings.output_pixel_format} requires even video dimensions, got "
            f"{plan.width}x{plan.height}"
        )
    frame_rate = f"{plan.fps_numerator}/{plan.fps_denominator}"
    command = [
        ffmpeg.executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{plan.width}x{plan.height}",
        "-framerate",
        frame_rate,
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        settings.video_codec,
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        settings.output_pixel_format,
    ]
    if settings.faststart:
        command.extend(("-movflags", "+faststart"))
    command.extend(settings.extra_output_args)
    command.extend(
        ("-frames:v", str(plan.frame_count), "-f", "mp4", str(temporary_output))
    )
    return tuple(command)


class _StderrCollector:
    def __init__(self, stream, *, limit: int = 65_536) -> None:
        self._stream = stream
        self._limit = limit
        self._data = bytearray()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _collect(self) -> None:
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                self._data.extend(chunk)
                if len(self._data) > self._limit:
                    del self._data[: len(self._data) - self._limit]
        except (OSError, ValueError):
            return

    def finish(self, timeout: float) -> str:
        self._thread.join(timeout)
        return bytes(self._data).decode("utf-8", errors="replace")


def encode_mp4_frames(
    frames: Iterable[_RgbFrame],
    plan: Mp4ExportPlan,
    output_path: str | Path,
    settings: Mp4ExportSettings = Mp4ExportSettings(),
    *,
    progress: ProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Mp4ExportResult:
    output = Path(output_path).expanduser()
    if output.suffix.casefold() != ".mp4":
        raise ValueError("MP4 output path must use the .mp4 extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not settings.overwrite:
        raise FileExistsError(f"MP4 output already exists: {output}")

    ffmpeg = probe_ffmpeg(settings.ffmpeg_executable)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.part")
    command = build_ffmpeg_mp4_command(ffmpeg, plan, temporary, settings)
    process = None
    collector = None
    frames_written = 0
    bytes_written = 0
    started = time.perf_counter()

    try:
        if cancellation_requested is not None and cancellation_requested():
            raise Mp4ExportCancelled("MP4 export cancelled before encoding started")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise Mp4ExportError(f"could not start FFmpeg: {exc}") from exc
        if process.stdin is None or process.stderr is None:
            raise Mp4ExportError(
                "FFmpeg process did not provide binary stdin/stderr pipes"
            )
        collector = _StderrCollector(process.stderr)
        collector.start()

        for frame in frames:
            if cancellation_requested is not None and cancellation_requested():
                raise Mp4ExportCancelled(
                    f"MP4 export cancelled after {frames_written} frames"
                )
            if frames_written >= plan.frame_count:
                raise Mp4ExportError(
                    "frame stream contains more than the planned "
                    f"{plan.frame_count} frames"
                )
            expected_index = frames_written
            if frame.index != expected_index:
                raise Mp4ExportError(
                    f"expected frame index {expected_index}, received {frame.index}"
                )
            expected_time = plan.time_seconds_text(expected_index)
            if frame.time_seconds_text != expected_time:
                raise Mp4ExportError(
                    f"frame {frame.index} has time {frame.time_seconds_text}s; "
                    f"expected {expected_time}s"
                )
            rgb = np.asarray(frame.rgb)
            expected_shape = (plan.height, plan.width, 3)
            if rgb.shape != expected_shape:
                raise Mp4ExportError(
                    f"frame {frame.index} has RGB shape {rgb.shape}; "
                    f"expected {expected_shape}"
                )
            if rgb.dtype != np.uint8:
                raise Mp4ExportError(
                    f"frame {frame.index} has dtype {rgb.dtype}; expected uint8 RGB"
                )
            payload = np.ascontiguousarray(rgb).tobytes(order="C")
            try:
                process.stdin.write(payload)
            except (BrokenPipeError, OSError) as exc:
                _close_stdin(process)
                return_code = _wait_process(
                    process, settings.shutdown_timeout_seconds
                )
                stderr = collector.finish(settings.shutdown_timeout_seconds)
                raise Mp4ExportError(
                    f"FFmpeg stopped while receiving frame {frame.index}",
                    return_code=return_code,
                    stderr=stderr,
                ) from exc
            frames_written += 1
            bytes_written += len(payload)
            if progress is not None:
                progress(
                    Mp4ExportProgress(
                        frames_written,
                        plan.frame_count,
                        frame.index,
                        frame.time_seconds_text,
                        bytes_written,
                    )
                )

        if frames_written != plan.frame_count:
            raise Mp4ExportError(
                f"frame stream ended after {frames_written} frames; "
                f"expected {plan.frame_count}"
            )
        if cancellation_requested is not None and cancellation_requested():
            raise Mp4ExportCancelled(
                f"MP4 export cancelled after {frames_written} frames"
            )
        _close_stdin(process)
        return_code = _wait_process(process, settings.shutdown_timeout_seconds)
        stderr = collector.finish(settings.shutdown_timeout_seconds)
        if return_code != 0:
            raise Mp4ExportError(
                f"FFmpeg exited with code {return_code}",
                return_code=return_code,
                stderr=stderr,
            )
        if not temporary.is_file():
            raise Mp4ExportError(
                "FFmpeg reported success but did not create the MP4 file"
            )
        if output.exists() and not settings.overwrite:
            raise FileExistsError(f"MP4 output appeared during export: {output}")
        os.replace(temporary, output)
        return Mp4ExportResult(
            output,
            frames_written,
            bytes_written,
            time.perf_counter() - started,
            ffmpeg,
            command,
        )
    except BaseException:
        if process is not None:
            _abort_process(process, settings.shutdown_timeout_seconds)
        if collector is not None:
            collector.finish(settings.shutdown_timeout_seconds)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _close_stdin(process) -> None:
    stream = getattr(process, "stdin", None)
    if stream is None or getattr(stream, "closed", False):
        return
    try:
        stream.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _wait_process(process, timeout: float) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=timeout)


def _abort_process(process, timeout: float) -> None:
    _close_stdin(process)
    try:
        running = process.poll() is None
    except OSError:
        running = False
    if running:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        try:
            process.kill()
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass
