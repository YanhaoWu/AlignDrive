"""Shared, streaming video encoding for simulation and perception renderers."""

from pathlib import Path
import subprocess
import tempfile

import cv2


def encode_video(source, destination):
    """Transcode an existing video to browser-compatible H.264."""
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(source),
            '-c:v',
            'libx264',
            '-profile:v',
            'high',
            '-crf',
            '20',
            '-pix_fmt',
            'yuv420p',
            str(destination),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def write_video(frames, output_path, fps):
    """Encode a stream of equally sized BGR frames without retaining them in memory."""
    if fps <= 0:
        raise ValueError('fps must be positive')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    expected_shape = None
    with tempfile.TemporaryDirectory(prefix='.render-', dir=str(output_path.parent)) as temporary:
        temporary_path = Path(temporary) / 'frames.mp4'
        try:
            for frame in frames:
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError('Expected a nonempty BGR image for every video frame')
                if expected_shape is None:
                    expected_shape = frame.shape
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(temporary_path),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (width + width % 2, height + height % 2),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot open video writer: {temporary_path}')
                elif frame.shape != expected_shape:
                    raise ValueError(
                        f'Inconsistent frame dimensions: {frame.shape} != {expected_shape}'
                    )
                height, width = frame.shape[:2]
                if height % 2 or width % 2:
                    frame = cv2.copyMakeBorder(
                        frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT
                    )
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
        if expected_shape is None:
            raise ValueError('No frames available for video encoding')
        encoded_path = Path(temporary) / 'encoded.mp4'
        encode_video(temporary_path, encoded_path)
        encoded_path.replace(output_path)
    return str(output_path)


def images_to_video(image_dir, output_path, fps=5):
    """Encode sorted debug images while ignoring non-image files."""
    image_paths = sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )

    def frames():
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f'Cannot read image: {image_path}')
            yield frame

    return write_video(frames(), output_path, fps)
