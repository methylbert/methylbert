# methylBERT

methylBERT is a deep learning model that leverages the BERT architecture, pretrained with masked DNA methylation levels. 
The model is implemented using PyTorch and PyTorch Lightning.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires the following packages in the `requirements.txt` file. You can install these packages using pip:

```bash
pip install -r requirements.txt
```

To install PyTorch, please follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). The installation command varies depending on your system configuration (OS, package manager, Python version, CUDA version). Here is an example for installing PyTorch on macOS with pip, Python 3.8 and CUDA 10.2:

```bash
pip install torch==1.7.1+cu102 torchvision==0.8.2+cu102 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Please replace the above command with the one that matches your system configuration.

## Code Structure
- `light/`: Contains the basic components for training and evaluation with pytorch lightning framework
- `methy/`: Main source code folder
  - `data/`: Contains the implementation of the data pipeline, including data loading and preprocessing
  - `model/`: Contains the implementation of the models
- `methy_light.py`: The script for training and evaluation
- `requirements.txt`: List of required dependencies
- `README.md`: This README file