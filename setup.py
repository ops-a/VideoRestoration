import os
from pathlib import Path

def create_project_structure():
    """Create the necessary directories for the project"""
    directories = [
        'data/raw/train',
        'data/raw/val',
        'data/raw/test',
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        'src/models',
        'src/data',
        'src/utils',
        'src/visualization',
        'notebooks',
        'tests',
        'logs',
        'checkpoints',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    create_project_structure()
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training data in 'data/raw/train'")
    print("2. Place your validation data in 'data/raw/val'")
    print("3. Place your test data in 'data/raw/test'")
    print("4. Run the training script: python src/train.py")
    print("5. Run the evaluation script: python src/evaluate.py") 