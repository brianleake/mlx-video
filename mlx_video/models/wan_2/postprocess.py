from pathlib import Path

import numpy as np


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Save video frames to an HEVC (hvc1) MP4 via Apple VideoToolbox.

    hevc_videotoolbox ignores libx264's `quality`/CRF knob, so we drive it with a
    generous bitrate. `-tag:v hvc1` makes the stream play in QuickTime / Photos /
    AVPlayer (the default `hev1` tag does not).

    Args:
        frames: Video frames [T, H, W, 3] uint8
        output_path: Output file path
        fps: Frames per second
    """
    h, w = int(frames.shape[1]), int(frames.shape[2])
    bitrate = max(10_000_000, int(w * h * fps * 1.5))  # generous; ~visually lossless
    try:
        import imageio

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="hevc_videotoolbox",
            format="FFMPEG",
            macro_block_size=None,
            output_params=["-tag:v", "hvc1", "-b:v", str(bitrate)],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        # imageio/VideoToolbox unavailable or failed → H.264 via OpenCV, then PNGs.
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except (ImportError, Exception):
            # Last resort: save as individual PNGs
            from PIL import Image

            out_dir = Path(output_path).parent / Path(output_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
            print(
                f"  (no video encoder available, saved {len(frames)} frames to {out_dir}/)"
            )
