import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Union, List, Tuple

class MediaDataset(Dataset):
    def __init__(self, 
                 distorted_path: str,
                 clean_path: str,
                 frame_size: Tuple[int, int] = (256, 256), 
                 transform=None):
        """
        Dataset for processing paired distorted and clean images
        Args:
            distorted_path: Path to directory containing distorted images
            clean_path: Path to directory containing clean (ground truth) images
            frame_size: Tuple of (height, width) for frame/image resizing
            transform: Optional transforms to apply to frames/images
        """
        self.distorted_path = distorted_path
        self.clean_path = clean_path
        self.frame_size = frame_size
        self.transform = transform
        self.pairs = []
        self._load_and_pair_images()

    def _load_and_pair_images(self):
        """Load and pair distorted and clean images"""
        # Get all image files from both directories
        distorted_files = sorted([
            f for f in os.listdir(self.distorted_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        clean_files = sorted([
            f for f in os.listdir(self.clean_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        # Create pairs of (distorted_path, clean_path)
        for distorted_file in distorted_files:
            # Extract base name without extension
            base_name = os.path.splitext(distorted_file)[0]
            # Create clean filename by appending "_clean" before extension
            clean_filename = f"{base_name}_clean{os.path.splitext(distorted_file)[1]}"
            
            # Check if clean file exists
            if clean_filename in clean_files:
                self.pairs.append((
                    os.path.join(self.distorted_path, distorted_file),
                    os.path.join(self.clean_path, clean_filename)
                ))
            else:
                print(f"Warning: No matching clean image found for {distorted_file}")

        if not self.pairs:
            raise ValueError("No valid image pairs found in the specified directories")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        distorted_path, clean_path = self.pairs[idx]
        
        # Load and preprocess distorted image
        distorted = cv2.imread(distorted_path)
        if distorted is None:
            raise ValueError(f"Failed to load distorted image: {distorted_path}")
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        distorted = cv2.resize(distorted, self.frame_size)
        
        # Load and preprocess clean image
        clean = cv2.imread(clean_path)
        if clean is None:
            raise ValueError(f"Failed to load clean image: {clean_path}")
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        clean = cv2.resize(clean, self.frame_size)
        
        if self.transform:
            distorted = self.transform(distorted)
            clean = self.transform(clean)
            
        # Convert to tensors and normalize
        distorted = torch.from_numpy(distorted).permute(2, 0, 1).float() / 255.0
        clean = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        
        return distorted, clean

class MediaProcessor:
    def __init__(self, input_path: Union[str, List[str]], 
                 output_path: str, 
                 frame_size: Tuple[int, int] = (256, 256),
                 is_video: bool = None):
        """
        Media processor for handling both video and image files
        Args:
            input_path: Path to input video/image or directory of images
            output_path: Path to save processed output
            frame_size: Tuple of (height, width) for frame/image resizing
            is_video: If None, automatically detect based on file extension
        """
        self.input_path = input_path
        self.output_path = output_path
        self.frame_size = frame_size
        self.is_video = is_video

    def extract_frames(self, output_dir: str):
        """Extract frames from video/images and save them"""
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(self.input_path, str):
            input_path = self.input_path
        else:
            input_path = self.input_path[0]

        # Auto-detect media type if not specified
        if self.is_video is None:
            self.is_video = self._is_video_file(input_path)

        if self.is_video:
            return self._extract_video_frames(output_dir)
        else:
            return self._extract_image_frames(output_dir)

    def _is_video_file(self, file_path: str) -> bool:
        """Check if the file is a video based on extension"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        return Path(file_path).suffix.lower() in video_extensions

    def _extract_video_frames(self, output_dir: str) -> int:
        """Extract frames from video"""
        cap = cv2.VideoCapture(self.input_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, self.frame_size)
            frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
        cap.release()
        return frame_count

    def _extract_image_frames(self, output_dir: str) -> int:
        """Extract and save frames from images"""
        if isinstance(self.input_path, str):
            if os.path.isdir(self.input_path):
                image_paths = [
                    os.path.join(self.input_path, f) 
                    for f in os.listdir(self.input_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ]
            else:
                image_paths = [self.input_path]
        else:
            image_paths = self.input_path

        frame_count = 0
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.resize(frame, self.frame_size)
                frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_count += 1

        return frame_count

    def create_video_from_frames(self, frames_dir: str, fps: int = 30):
        """Create video from processed frames"""
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        if not frames:
            raise ValueError("No frames found in the specified directory")

        first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame_path = os.path.join(frames_dir, frame)
            frame_data = cv2.imread(frame_path)
            out.write(frame_data)

        out.release()

    def process_media(self, model, device='cuda'):
        """Process media (video/images) through the model"""
        if isinstance(self.input_path, str):
            input_path = self.input_path
        else:
            input_path = self.input_path[0]

        # Auto-detect media type if not specified
        if self.is_video is None:
            self.is_video = self._is_video_file(input_path)

        if self.is_video:
            self._process_video(model, device)
        else:
            self._process_images(model, device)

    def _process_video(self, model, device):
        """Process video frames through the model"""
        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)
        
        model.eval()
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Preprocess frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(device)
                
                # Process frame
                restored = model(frame_tensor)
                
                # Convert back to video frame
                restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored = (restored * 255).astype(np.uint8)
                restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                
                out.write(restored)
        
        cap.release()
        out.release()

    def _process_images(self, model, device):
        """Process individual images through the model"""
        if isinstance(self.input_path, str):
            if os.path.isdir(self.input_path):
                image_paths = [
                    os.path.join(self.input_path, f) 
                    for f in os.listdir(self.input_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ]
            else:
                image_paths = [self.input_path]
        else:
            image_paths = self.input_path

        model.eval()
        with torch.no_grad():
            for img_path in image_paths:
                # Read and preprocess image
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(device)
                
                # Process image
                restored = model(frame_tensor)
                
                # Convert back to image
                restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored = (restored * 255).astype(np.uint8)
                restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                
                # Save processed image
                output_path = os.path.join(
                    os.path.dirname(self.output_path),
                    'restored_' + os.path.basename(img_path)
                )
                cv2.imwrite(output_path, restored)