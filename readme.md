# DAGNet: A Dual-View Attention-Guided Network for Efficient X-ray Security Inspection

## Requirements
The following dependencies are required to run the code:

- Python 3.x
- PyTorch 2.5.1
- numpy
- opencv-python
- scikit-learn
- scipy
- einops
- thop
- torchmetrics
- PyWavelets
- timm
- pandas
- openpyxl
- transformers>=4.5.0
- dill

You can install them using the following:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Dataset Preparation
First, download and extract the dataset from the following link:

- [DvXray Dataset](https://github.com/Mbwslib/DvXray)

After downloading, extract the dataset to the `data/` directory in the root of the project.

### 2. Dataset Splitting
To split the dataset into training, validation, and test sets, use the `split_dataset.py` script. This script will automatically divide the dataset with a 7:2:1 ratio:

```
python split_dataset.py
```

### 3. Training the Model
Before training, ensure all dependencies are installed, and the necessary parameters are configured. You can start training by running the `train.py` script. :

```
python train.py
```

or
```
torchrun --nproc_per_node=2 train.py
```

### 4. Model Evaluation
After training, you can evaluate the model on the test set using the following command:

```
python train.py --eval -r <checkpoint>
```

## Citation
If you use this code or the method presented in the paper, please cite the following:

TODO: Add citation information here.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.