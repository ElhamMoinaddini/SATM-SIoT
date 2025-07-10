## Overview
**A Self-Adaptive Model for Evaluating Trust in Social Internet of Things Using Machine Learning**

This research proposes and implements a self-adaptive trust management model for SIoT, named SATM-SIoT, which addresses these security and operational challenges. The proposed model employs a hybrid cloud-fog architecture, delegating trust computations to fog nodes while the cloud layer utilizes federated learning to update and aggregate machine learning models. By integrating multidimensional metrics—including social relationships and service quality—the model provides a comprehensive trust evaluation and employs a fuzzy system to enhance feedback accuracy. Leveraging fog-layer computations, the model remains resource-efficient and scalable. To manage environmental dynamics, it combines historical behavior analysis, federated learning for continuous model updates, and a MAPE-K control loop for autonomous monitoring and adaptation.

For detailed explanations, refer to https://www.sciencedirect.com/science/article/abs/pii/S1389128625001550


## Installation

### Prerequisites
python 3.9.16

tensorflow 2.10.1

scikit-learn 1.1.1

### Clone the repo
```bash
git clone https://github.com/ElhamMoinaddini/SATM-SIoT.git
cd SATM-SIoT
```

### Create a New Conda Environment with Python 3.9
```bash
conda create -n satm-siots python=3.9 -y

conda activate satm-siots
```

### Install TensorFlow and scikit-learn  

#### 1. Create a new conda environment (optional but recommended)
```bash
conda create -n tf_sklearn_env python=3.9 -y
```
#### 2. Activate the environment
```bash
conda activate tf_sklearn_env
```
#### 3. Install TensorFlow from conda-forge
```bash
conda install -c conda-forge tensorflow=2.10.1 -y
```
#### 4. Install Scikit-Learn
```bash
conda install -c conda-forge scikit-learn=1.1.1 -y
```
#### 5. Verify installations
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sklearn; print(f'Scikit-Learn: {sklearn.__version__}')"
```

#### Run SATM-SIoT Simulation
```bash
python main.py -t 5000 -p cloud -nf 5 -nd 5
```
#### Arguments:
-t 5000: Number of rounds

-p cloud: Cloud or edge, useing Cloud is recommend for this simulation

-nf 5: Number of fog nodes

-nd 5: Initial number of devices in each fog nodes

## Disclaimer
This is academic research project provided "as-is" without warranties. Users assume all responsibility for implementation decisions.

## Acknowledgments
1- The code of the YAFS (Yet Another Fog Simulator) is inspired https://github.com/acsicuib/YAFS by Isaac Lera

2- The code of the Flower: A Friendly Federated AI Framework is inspired https://github.com/adap/flower
