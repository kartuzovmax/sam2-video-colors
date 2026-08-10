# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import io
import time
import torch
import mimetypes
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv
from typing import Iterator
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path
from contextlib import contextmanager
import shutil
import tempfile

mimetypes.add_type("image/webp", ".webp")


DEVICE = "cuda"
MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/sam-2/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        global build_sam2_video_predictor

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print("sam2 not found. Installing...")
            os.system("pip install --no-build-isolation --break-system-packages -e .")
            from sam2.build_sam import build_sam2_video_predictor

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["sam2_hiera_large.pt"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = f"{MODEL_CACHE}/sam2_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.mask_annotator = sv.MaskAnnotator()
        self.box_annotator = sv.BoxAnnotator()

    def parse_inputs(
        self,
        click_coordinates: str,
        click_labels: str,
        click_frames: str,
        click_object_ids: str,
    ) -> tuple:
        click_coordinates = click_coordinates.replace(" ", "")
        click_labels = click_labels.replace(" ", "")
        click_frames = click_frames.replace(" ", "")
        click_object_ids = click_object_ids.replace(" ", "")

        # Parse click coordinates
        click_list = [
            list(map(int, click.replace(" ", "").split(",")))
            for click in click_coordinates.strip("[]").split("],[")
        ]
        num_clicks = len(click_list)

        # Handle click labels
        click_labels_list = list(map(int, click_labels.split(",")))
        click_labels_list = click_labels_list * (
            num_clicks // len(click_labels_list) + 1
        )
        click_labels_list = click_labels_list[:num_clicks]

        # Handle click frames
        click_frames_list = (
            list(map(int, click_frames.split(",")))
            if click_frames
            else [0] * num_clicks
        )
        click_frames_list = click_frames_list * (
            num_clicks // len(click_frames_list) + 1
        )
        click_frames_list = click_frames_list[:num_clicks]

        # Handle click object IDs
        if click_object_ids:
            object_ids_list = click_object_ids.split(",")
        else:
            object_ids_list = [f"object_{i}" for i in range(1, num_clicks + 1)]
        object_ids_list = object_ids_list * (num_clicks // len(object_ids_list) + 1)
        object_ids_list = object_ids_list[:num_clicks]

        # Map string labels to unique integer IDs
        label_to_id = {}
        id_counter = 1
        object_ids_int_list = []
        for label in object_ids_list:
            if label not in label_to_id:
                label_to_id[label] = id_counter
                id_counter += 1
            object_ids_int_list.append(label_to_id[label])

        return (click_list, click_labels_list, click_frames_list, object_ids_int_list)

    def save_image(self, image: Image.Image, path: Path, format: str, quality: int):
        save_params = {"format": format.upper()}
        if format.lower() != "png":
            save_params["quality"] = quality
            save_params["optimize"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **save_params)

    @contextmanager
    def temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def predict(
        self,
        input_video: Path = Input(description="Input video file path"),
        # Segmentation inputs
        click_coordinates: str = Input(
            description="Click coordinates as '[x,y],[x,y],...'. Determines number of clicks."
        ),
        click_labels: str = Input(
            description="Click types (1=foreground, 0=background) as '1,1,0,1'. Auto-extends if shorter than coordinates.",
            default="1",
        ),
        click_frames: str = Input(
            description="Frame indices for clicks as '0,0,150,0'. Auto-extends if shorter than coordinates.",
            default="0",
        ),
        click_object_ids: str = Input(
            description="Object labels for clicks as 'person,dog,cat'. Auto-generates if missing or incomplete.",
            default="",
        ),
        # Output type
        mask_type: str = Input(
            default="binary",
            choices=["binary", "highlighted", "greenscreen"],
            description="Mask type: binary (B&W), highlighted (colored overlay), or greenscreen",
        ),
        annotation_type: str = Input(
            default="mask",
            choices=["mask", "box", "both"],
            description="Annotation type: mask only, bounding box only, or both (ignored for binary and greenscreen)",
        ),
        # Output format
        output_video: bool = Input(
            default=False,
            description="True for video output, False for image sequence",
        ),
        # Video-specific options
        video_fps: int = Input(
            description="Video output frame rate (ignored for image sequence)",
            default=30,
            ge=1,
            le=60,
        ),
        # Image sequence-specific options
        output_format: str = Input(
            description="Image format for sequence (ignored for video)",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="JPG/WebP compression quality (0-100, ignored for PNG and video)",
            default=80,
            ge=0,
            le=100,
        ),
        # Background color for greenscreen mode
        background_color: str = Input(
            description="Background color for greenscreen mode",
            default="green",
            choices=["green", "red", "blue", "yellow", "cyan", "magenta", "white", "black"],
        ),
        # General output option
        output_frame_interval: int = Input(
            default=1,
            description="Output every Nth frame. 1=all frames, 2=every other, etc.",
        ),
    ) -> Iterator[Path]:
        # Parse inputs
        click_list, click_labels_list, click_frames_list, object_ids_int_list = (
            self.parse_inputs(
                click_coordinates, click_labels, click_frames, click_object_ids
            )
        )

        # Create output directory
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        with self.temporary_directory() as frame_directory_path:
            # Extract video frames
            video_info = sv.VideoInfo.from_video_path(str(input_video))
            frames_generator = sv.get_video_frames_generator(str(input_video))
            frames_sink = sv.ImageSink(
                target_dir_path=str(frame_directory_path),
                image_name_pattern="{:05d}.jpeg",
            )

            with frames_sink:
                for frame in tqdm(
                    frames_generator,
                    total=video_info.total_frames,
                    desc="Splitting video into frames",
                ):
                    frames_sink.save_image(frame)

            # Initialize SAM predictor
            inference_state = self.predictor.init_state(
                video_path=str(frame_directory_path)
            )

            # Group clicks by (obj_id, frame) and pass all points together
            # This is critical: passing points one-at-a-time gives different (worse) results
            from collections import defaultdict
            grouped = defaultdict(lambda: {"points": [], "labels": []})
            for click, click_type, frame, obj_id in zip(
                click_list, click_labels_list, click_frames_list, object_ids_int_list
            ):
                key = (obj_id, frame)
                grouped[key]["points"].append(click)
                grouped[key]["labels"].append(click_type)

            for (obj_id, frame), data in grouped.items():
                points = np.array(data["points"], dtype=np.float32)
                labels = np.array(data["labels"], dtype=np.int32)
                _, _, _ = self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            # Propagate masks
            masks_generator = self.predictor.propagate_in_video(inference_state)

            # Generate and yield results
            frames_generator = sv.get_video_frames_generator(str(input_video))

            if output_video:
                video_path = output_dir / "output_video.mp4"
                frame_paths = []

                for frame_idx, (frame, (_, tracker_ids, mask_logits)) in enumerate(
                    zip(frames_generator, masks_generator)
                ):
                    if frame_idx % output_frame_interval != 0:
                        continue

                    annotated_frame = self.process_frame(
                        frame, mask_logits, tracker_ids, mask_type, annotation_type, background_color
                    )
                    frame_path = frame_directory_path / f"frame_{frame_idx:05d}.png"
                    Image.fromarray(annotated_frame).save(frame_path)
                    frame_paths.append(frame_path)

                # Use FFmpeg to create the video
                first_frame = Image.open(frame_paths[0])
                frame_width, frame_height = first_frame.size

                ffmpeg_command = (
                    f"ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 "
                    f"-s {frame_width}x{frame_height} -r {video_fps} "
                    f"-i - -c:v h264_nvenc -preset p7 -tune hq "
                    f"-rc vbr -cq 10 -qmin 10 -qmax 20 "
                    f"-b:v 0 -maxrate:v 200M -bufsize 200M "
                    f"-profile:v high -pix_fmt yuv420p "
                    f"-gpu 0 -rc-lookahead 32 -temporal-aq 1 -aq-strength 15 "
                    f"{video_path}"
                )
                ffmpeg_process = subprocess.Popen(
                    ffmpeg_command.split(),
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                for frame_path in frame_paths:
                    with Image.open(frame_path) as img:
                        frame_data = np.array(img.convert("RGB"))
                        # Convert BGR to RGB
                        frame_data = frame_data[:, :, ::-1]
                        ffmpeg_process.stdin.write(frame_data.tobytes())

                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                # Clean up temporary frame files
                for frame_path in frame_paths:
                    frame_path.unlink()

                yield video_path

            else:
                for frame_idx, (frame, (_, tracker_ids, mask_logits)) in enumerate(
                    zip(frames_generator, masks_generator)
                ):
                    if frame_idx % output_frame_interval != 0:
                        continue

                    annotated_frame = self.process_frame(
                        frame, mask_logits, tracker_ids, mask_type, annotation_type, background_color
                    )
                    output_path = output_dir / f"frame_{frame_idx:05d}.{output_format}"
                    self.save_image(
                        Image.fromarray(annotated_frame),
                        output_path,
                        output_format,
                        output_quality,
                    )
                    yield output_path

    def process_frame(
        self, frame, mask_logits, tracker_ids, mask_type, annotation_type, background_color="green"
    ):
        colors = {
            "green": [0, 255, 0],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255],
            "white": [255, 255, 255],
            "black": [0, 0, 0],
        }

        masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
        if len(masks.shape) == 4:
            masks = np.squeeze(masks, axis=1)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array(tracker_ids),
        )

        if mask_type == "highlighted":
            annotated_frame = frame.copy()
            if annotation_type in ["mask", "both"]:
                annotated_frame = self.mask_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
            if annotation_type in ["box", "both"]:
                annotated_frame = self.box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
        elif mask_type == "binary":
            mask = masks.any(axis=0)
            if masks.shape[0] > 1:
                mask = self.close_union_gaps(mask)
            annotated_frame = (mask * 255).astype(np.uint8)
        elif mask_type == "greenscreen":
            color_rgb = colors.get(background_color.lower(), [0, 255, 0])
            bg = np.full(frame.shape, color_rgb, dtype=np.uint8)
            mask = masks.any(axis=0)
            if masks.shape[0] > 1:
                mask = self.close_union_gaps(mask)
            annotated_frame = np.where(mask[..., None], frame, bg)

        return annotated_frame

    def close_union_gaps(self, mask):
        """Fill the thin uncovered band SAM2 leaves where two tracked objects touch
        or overlap (e.g. a person and the microphone in front of them). Each object's
        mask is thresholded tightly, so the union renders a 2-10px background-colored
        divider line/ring at the contact. Morphological closing bridges only gaps
        narrower than the kernel, leaving outer silhouettes intact. Only called for
        multi-object jobs, so single-subject output stays bit-identical."""
        import cv2

        h, w = mask.shape
        k = max(5, int(round(min(h, w) * 0.012)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed.astype(bool)
