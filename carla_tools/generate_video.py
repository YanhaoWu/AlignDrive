import os
import json
import argparse
from pathlib import Path


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    import cv2
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()
    if not images:
        raise ValueError(f'No images found in {images_folder}/rgb_front')

    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    if frame is None:
        raise ValueError(f'Cannot read image: {images[0]}')
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not video.isOpened():
        raise RuntimeError(f'Cannot create video: {output_video}')

    try:
        for index in range(len(images)):
            image = images[index]
            frame_id = Path(image).stem
            meta_path = Path(images_folder) / 'meta' / f'{frame_id}.json'
            if not meta_path.is_file() and frame_id.isdigit():
                meta_path = Path(images_folder) / 'meta' / f'{int(frame_id):04}.json'
            with meta_path.open() as metadata_file:
                meta = json.load(metadata_file)
            steer = float(meta['steer'])
            throttle = float(meta['throttle'])
            brake = float(meta['brake'])
            speed = float(meta['speed'])
            text = f'speed: {speed:.2f}, steer: {steer:.2f}, throttle: {throttle:.2f}, brake: {brake:.2f}'
            img = cv2.imread(os.path.join(images_folder, 'rgb_front', image))
            if img is None:
                raise ValueError(f'Cannot read image: {image}')
            if img.shape[:2] != (height, width):
                raise ValueError(f'Inconsistent frame size: {image}')
            cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            video.write(img)
    finally:
        video.release()

def main():
    parser = argparse.ArgumentParser(description='Create an RGB video from rgb_front images and meta JSON files')
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error('fps must be positive')
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    create_video(str(args.scene_dir.expanduser().resolve()), str(output), args.fps,
                 1, (255, 255, 255), (50, 50))
    print(output)


if __name__ == '__main__':
    main()
