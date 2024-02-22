"""
Utilities for loading extracting frames from case-level videos.
BY: MDOROSAN 26112023

SAMPLE_SCRIPT : Ensure environment has necessary packages

```
python utils/extract_frames.py --src_dir '/home/mdorosan/2023/kvasir-capsule-dataset/labelled_videos' --dst_dir '/home/mdorosan/2023/kvasir-capsule-dataset/extracted_labelled_images'
```

FUTURE UPDATES:
* Line by line metadata/log update
* Add verbose option
"""

import os
import cv2
import time
import argparse
from tqdm import tqdm
import json


def get_video_paths(src_dir, fext=".mp4"):
    """Return a list of filepaths to files corresponding to fext"""
    fplist = []
    for path, subdir, files in os.walk(src_dir):
        if files:
            for name in files:
                if name.endswith(fext):
                    fplist.append(os.path.join(path, name))
    return fplist


def extract_id(video_file):
    """Return patient_id (case id) from video_file path"""
    return video_file.split("/")[-1].split(".")[0]


def extract_frames(video_file, dst_dir, save_fext=".jpg", subsample_fps=None):
    """Extract frames of a video file and save to dst_dir."""
    # save in dst_dir, case_id/video_name as subdir
    video_id = extract_id(video_file)
    output_path = os.path.join(dst_dir, video_id)
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    
    # frames per second
    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_dims = (frame_width, frame_height)

    metadata = {
        "video_id": video_id,
        "filesize (MB)": os.path.getsize(video_file) / 1e6,
        "frame_count": frame_count,
        "default_fps": video_frame_rate,
        "subsample_fps": subsample_fps,
        "frame_dims_wxh": frame_dims,
    }
    
    frame_step = 1
    if subsample_fps:
        if subsample_fps > default_fps:
            raise ValueError(f"Cannot extract at {subsample_fps} FPS, video FPS is {default_fps}")
        frame_step = subsample_fps
        
    corrupted_frames = []    
    for frame_num in tqdm(range(0, frame_count - 1, frame_step)):
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num) # frame locator by id
        # cv2.CAP_PROP_POS_MSEC # also a locator using feed time in milliseconds
        if not cap.isOpened():
            raise RuntimeError("Feed is closed.")
            
        ret, frame = cap.read()
        if not ret:
            corrupted_frames.append(frame_num)
            # raise RuntimeError(f"Error reading frame {frame_num}.")
        else:
            cv2.imwrite(
                    os.path.join(output_path, f"{video_id}_{frame_num}{save_fext}"),
                    frame
            )
    metadata["extracted_count"] = frame_count - len(corrupted_frames)
    metadata["corrupted_frames"] = corrupted_frames
    cap.release()
    return metadata

    ## while loop version, no subsample option
    
    # i = 0
    # corrupted_frames = []      
    # while cap.isOpened():
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     ret, frame = cap.read()
    #     if not ret:
    #         corrupted_frames.append(i)
    #     else:
    #         cv2.imwrite(
    #             os.path.join(output_path, f"{video_id}_{i}{save_fext}"),
    #             frame
    #         )
    #     i += 1
    #     if i == frame_count:
    #         break
    
def _save_metadata(records, dst_dir):
    meta_path = os.path.join(dst_dir, "metadata.json")
    with open(meta_path, "w") as file:
        json.dump(records, file)
    pass


def mass_extract(
        src_dir, dst_dir, path_collector, subsample_fps=None, **kwargs):
    """Mass extract all videos from src_dir, save to dst_dir"""
    fplist = path_collector(src_dir, **kwargs)
    records = []
    # loop extract_frames
    for video_file in fplist:
        metadata = extract_frames(
            video_file, dst_dir, save_fext=".jpg", subsample_fps=subsample_fps)
        records.append(metadata)
    _save_metadata(records, dst_dir)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        required=True,
        help='Name of source directory that contains .gvf files'
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        required=True,
        help='Destination directory for .jpg frames'
    )
    parser.add_argument(
        '--subsample_fps',
        type=int,
        default=None,
        help='Number of frames to extract per second, defaults to video FPS'
    )
    args = parser.parse_args()
    start = time.time()
    mass_extract(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        path_collector=get_video_paths,
        subsample_fps=args.subsample_fps,
    )
    end = time.time()
    elapsed_min = (end - start) / 60
    print(f"DONE  in {elapsed_min:.4f} minutes")
