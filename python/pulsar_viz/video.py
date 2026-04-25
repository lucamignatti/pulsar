from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import time


_SCREEN_DEVICE_RE = re.compile(r"\[(?P<index>\d+)\]\s+Capture screen")
_CSV_INTS_RE = re.compile(r"-?\d+")
_DEFAULT_INPUT_PIXEL_FORMATS = ("bgr0", "nv12", "uyvy422", "yuyv422")


@dataclass(frozen=True, slots=True)
class WindowBounds:
    x: int
    y: int
    width: int
    height: int


class RLViserVideoRecorder:
    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        app_names: tuple[str, ...] = ("rlviser", "pyviser"),
        startup_timeout_seconds: float = 10.0,
        input_pixel_formats: tuple[str, ...] = _DEFAULT_INPUT_PIXEL_FORMATS,
    ):
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Recording requires `ffmpeg` to be installed and available on PATH.")

        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.app_names = app_names
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.input_pixel_formats = input_pixel_formats
        self._ffmpeg = ffmpeg
        self._process: subprocess.Popen[bytes] | None = None
        self._started = False
        self._active_input_pixel_format: str | None = None

    def start(self) -> None:
        if self._started:
            return

        screen_index = self._detect_screen_device_index()
        screen_scale, capture_size = self._detect_screen_scale(screen_index)
        bounds = self._wait_for_window_bounds()
        crop = self._crop_filter(bounds, screen_scale, capture_size)
        self._bring_window_to_front()
        self._process = self._start_ffmpeg(screen_index, crop)
        self._started = True

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        self._process = None
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.write(b"q\n")
                process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()

        try:
            return_code = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=5.0)

        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        if return_code != 0:
            details = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(self._ffmpeg_error(f"ffmpeg exited with status {return_code}", details))

    def _start_ffmpeg(self, screen_index: int, crop_filter: str) -> subprocess.Popen[bytes]:
        last_error = ""
        for input_pixel_format in self.input_pixel_formats:
            process = subprocess.Popen(
                self._build_ffmpeg_command(screen_index, crop_filter, input_pixel_format),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.35)
            if process.poll() is None:
                self._active_input_pixel_format = input_pixel_format
                return process

            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
            process.wait(timeout=1.0)
            details = stderr.decode("utf-8", errors="replace").strip()
            last_error = details

        raise RuntimeError(
            self._ffmpeg_error(
                "ffmpeg failed to start screen capture with every supported input pixel format",
                f"Tried {', '.join(self.input_pixel_formats)}. Last ffmpeg output: {last_error}",
            )
        )

    def _build_ffmpeg_command(self, screen_index: int, crop_filter: str, input_pixel_format: str) -> list[str]:
        return [
            self._ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-capture_cursor",
            "0",
            "-pixel_format",
            input_pixel_format,
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            f"{screen_index}:none",
            "-an",
            "-vf",
            crop_filter,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "18",
            str(self.output_path),
        ]

    def _detect_screen_device_index(self) -> int:
        result = subprocess.run(
            [self._ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            check=False,
            capture_output=True,
            text=True,
        )
        device_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        indices = [int(match.group("index")) for match in _SCREEN_DEVICE_RE.finditer(device_output)]
        if not indices:
            raise RuntimeError(
                "Unable to find an AVFoundation screen capture device. Grant screen recording permission to the "
                "terminal or Python process and ensure a display is available."
            )
        return min(indices)

    def _detect_screen_scale(self, screen_index: int) -> tuple[tuple[float, float], tuple[int, int]]:
        capture_width, capture_height = self._capture_dimensions(screen_index)
        if capture_width <= 0 or capture_height <= 0:
            return ((1.0, 1.0), (capture_width, capture_height))

        try:
            logical_width, logical_height = self._desktop_bounds()
        except Exception:
            return ((1.0, 1.0), (capture_width, capture_height))
        if logical_width <= 0 or logical_height <= 0:
            return ((1.0, 1.0), (capture_width, capture_height))

        return ((capture_width / logical_width, capture_height / logical_height), (capture_width, capture_height))

    def _desktop_bounds(self) -> tuple[int, int]:
        raw = self._run_osascript(
            'tell application "Finder" to get bounds of window of desktop as string'
        )
        values = [int(token) for token in _CSV_INTS_RE.findall(raw)]
        if len(values) != 4:
            raise RuntimeError(f"Unable to parse desktop bounds from AppleScript output: {raw!r}")
        left, top, right, bottom = values
        return (right - left, bottom - top)

    def _capture_dimensions(self, screen_index: int) -> tuple[int, int]:
        cmd = [
            self._ffmpeg,
            "-f",
            "avfoundation",
            "-framerate",
            "1",
            "-i",
            f"{screen_index}:none",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        match = re.search(r"(\d{2,5})x(\d{2,5})", combined)
        if match is None:
            raise RuntimeError(
                "Unable to determine capture dimensions from ffmpeg screen input. "
                "Grant screen recording permission and try again."
            )
        return (int(match.group(1)), int(match.group(2)))

    def _wait_for_window_bounds(self) -> WindowBounds:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            for app_name in self.app_names:
                try:
                    bounds = self._window_bounds(app_name)
                except Exception as exc:
                    last_error = exc
                    continue
                if bounds is not None:
                    return bounds
            time.sleep(0.1)

        if last_error is not None:
            raise RuntimeError(
                "Timed out waiting for the RLViser window to appear. Screen recording on macOS also requires "
                "Accessibility permission for window inspection."
            ) from last_error
        raise RuntimeError("Timed out waiting for the RLViser window to appear.")

    def _window_bounds(self, app_name: str) -> WindowBounds | None:
        raw = self._run_osascript(
            f'''
            tell application "System Events"
                if not (exists application process "{app_name}") then
                    return ""
                end if
                tell application process "{app_name}"
                    if (count of windows) is 0 then
                        return ""
                    end if
                    set theWindow to front window
                    set thePos to position of theWindow
                    set theSize to size of theWindow
                    return (item 1 of thePos as string) & "," & (item 2 of thePos as string) & "," & (item 1 of theSize as string) & "," & (item 2 of theSize as string)
                end tell
            end tell
            '''
        )
        values = [int(token) for token in _CSV_INTS_RE.findall(raw)]
        if len(values) != 4:
            return None
        return WindowBounds(x=values[0], y=values[1], width=values[2], height=values[3])

    def _bring_window_to_front(self) -> None:
        for app_name in self.app_names:
            try:
                self._run_osascript(
                    f'''
                    tell application "System Events"
                        if exists application process "{app_name}" then
                            tell application process "{app_name}" to set frontmost to true
                            return "ok"
                        end if
                    end tell
                    return ""
                    '''
                )
                return
            except Exception:
                continue

    def _crop_filter(
        self,
        bounds: WindowBounds,
        scale: tuple[float, float],
        capture_size: tuple[int, int],
    ) -> str:
        scale_x, scale_y = scale
        capture_width, capture_height = capture_size
        crop_x = self._even_floor(bounds.x * scale_x)
        crop_y = self._even_floor(bounds.y * scale_y)
        crop_w = self._even_floor(bounds.width * scale_x)
        crop_h = self._even_floor(bounds.height * scale_y)

        if crop_x >= capture_width or crop_y >= capture_height:
            raise RuntimeError(f"RLViser window is outside the captured screen bounds: {bounds}")

        crop_w = min(crop_w, capture_width - crop_x)
        crop_h = min(crop_h, capture_height - crop_y)
        crop_w = self._even_floor(crop_w)
        crop_h = self._even_floor(crop_h)

        if crop_w <= 0 or crop_h <= 0:
            raise RuntimeError(f"Invalid RLViser window bounds for recording: {bounds}")

        return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},format=yuv420p"

    @staticmethod
    def _even_floor(value: float) -> int:
        integer = max(0, int(value))
        return integer if integer % 2 == 0 else integer - 1

    def _run_osascript(self, script: str) -> str:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(self._ffmpeg_error("osascript failed while querying the RLViser window", stderr))
        return result.stdout.strip()

    def _ffmpeg_error(self, summary: str, details: str | None = None) -> str:
        message = f"{summary}. Output path: {self.output_path}"
        if details:
            message = f"{message}. Details: {details}"
        return message
