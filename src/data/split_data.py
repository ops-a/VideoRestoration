import os
import shutil
import random
from pathlib import Path
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def split_data(train_distorted_dir: str, train_clean_dir: str, 
               val_distorted_dir: str, val_clean_dir: str,
               val_split: float = 0.1):
    """
    Split training data into training and validation sets.
    
    Args:
        train_distorted_dir: Directory containing distorted training images
        train_clean_dir: Directory containing clean training images
        val_distorted_dir: Directory to store distorted validation images
        val_clean_dir: Directory to store clean validation images
        val_split: Fraction of data to use for validation (default: 0.1)
    """
    # Create all directories if they don't exist
    for dir_path in [train_distorted_dir, train_clean_dir, val_distorted_dir, val_clean_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    # Get list of distorted images
    distorted_images = [f for f in os.listdir(train_distorted_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not distorted_images:
        raise ValueError(f"No images found in {train_distorted_dir}")
    
    logging.info(f"Found {len(distorted_images)} images in {train_distorted_dir}")
    
    # Calculate number of images to move to validation set
    num_val = int(len(distorted_images) * val_split)
    if num_val < 1:
        num_val = 1  # Ensure at least one validation image
    
    # Randomly select images for validation
    val_images = random.sample(distorted_images, num_val)
    
    # Move selected images to validation directories
    for img in val_images:
        # Get base name without extension
        base_name = os.path.splitext(img)[0]
        clean_img = f"{base_name}_clean{os.path.splitext(img)[1]}"
        
        # Check if clean image exists
        clean_src = os.path.join(train_clean_dir, clean_img)
        if not os.path.exists(clean_src):
            logging.warning(f"Clean image {clean_img} not found for {img}")
            continue
        
        # Move distorted image
        src_distorted = os.path.join(train_distorted_dir, img)
        dst_distorted = os.path.join(val_distorted_dir, img)
        shutil.move(src_distorted, dst_distorted)
        
        # Move clean image
        dst_clean = os.path.join(val_clean_dir, clean_img)
        shutil.move(clean_src, dst_clean)
        
        logging.info(f"Moved {img} and {clean_img} to validation set")

if __name__ == "__main__":
    setup_logging()
    
    # Get paths from config
    config_path = Path(__file__).parent.parent.parent / "config" / "training_config.yaml"
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise
    
    # Get directory paths from config
    try:
        train_distorted = config['data']['train_distorted']
        train_clean = config['data']['train_clean']
        val_distorted = config['data']['val_distorted']
        val_clean = config['data']['val_clean']
    except KeyError as e:
        logging.error(f"Missing required configuration key: {e}")
        raise
    
    # Split the data
    try:
        split_data(train_distorted, train_clean, val_distorted, val_clean)
        logging.info("Data splitting completed successfully")
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise 