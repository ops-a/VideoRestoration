# Restoration of Distorted Video Frames

> Problem Statement: Develop an ML system to restore video frames distorted by atmospheric conditions.
Problem Description: Atmospheric distortions degrade video quality. An ML model using regression and feature extraction can analyze frame sequences, apply corrective transformations, and enhance clarity, ensuring better visual quality

---

### Datasets
The dataset can be downloaded from: [Rain Drop Data](https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K)
- Put the data in a directory named datasets inside the RainRemoval directory.

### Install the dependencies:

```pip install -r requirements.txt```

### Run the main file using the following command. The command line arguments are optional. The below mentioned values are default:

```python main.py --image_size 256 --epochs 20 --batch_size 8 --learning_rate 0.0001```


### Test the model (again image size is optional):
`python -m src.test --image_size 256`