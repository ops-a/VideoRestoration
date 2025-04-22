# Video Restoration Project

This project implements various image and video restoration models, including:
- VideoRestorationModel (Residual Block architecture)
- EncoderDecoderModel
- RainRemovalModel (with SEBlock)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd VideoRestoration
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
VideoRestoration/
├── checkpoints/           # Model checkpoints
├── config/               # Configuration files
├── data/                 # Data processing scripts
├── src/                  # Source code
│   ├── models/          # Model definitions
│   ├── utils/           # Utility functions
│   └── evaluate.py      # Evaluation script
├── web/                 # Web interface
│   ├── static/          # Static files (CSS, JS, uploads)
│   └── templates/       # HTML templates
└── requirements.txt     # Project dependencies
```

## Model Evaluation

To evaluate a model using a checkpoint:

```bash
python3 src/evaluate.py --checkpoint checkpoints/advanced_rain_removal_net.pth --config config/eval_config.yaml
```

The evaluation script supports multiple model architectures and will automatically detect the correct one based on the checkpoint.

## Web Interface

To run the web interface:

```bash
python3 web/app.py
```

The web interface will be available at `http://localhost:5000`

Features:
- Image upload and processing
- Real-time restoration
- Quality metrics (PSNR, SSIM)
- Support for multiple model architectures

## Model Architectures

The project supports three model architectures:

1. **VideoRestorationModel**
   - Residual block architecture
   - Good for general video restoration

2. **EncoderDecoderModel**
   - Encoder-decoder structure
   - Suitable for complex restoration tasks

3. **RainRemovalModel**
   - Includes SEBlock for attention
   - Specifically designed for rain removal

## Configuration

The evaluation configuration is specified in `config/eval_config.yaml`. Example:

```yaml
model:
  in_channels: 3
  num_blocks: 16

data:
  test_dir: "data/test"
  batch_size: 1
```

## Troubleshooting

1. **Model Loading Issues**
   - Ensure the checkpoint file exists in the correct location
   - Check that the model architecture matches the checkpoint

2. **Web Interface Issues**
   - Make sure the uploads directory exists: `web/static/uploads`
   - Check file permissions
   - Ensure all dependencies are installed

3. **Evaluation Errors**
   - Verify the test data directory structure
   - Check configuration file paths
   - Ensure sufficient disk space for processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]