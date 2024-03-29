# VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain

Welcome to the PyTorch Lightning implementation of VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain ([ICML'2020](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)).
This project is an unofficial implementation and was developed to mirror the methods described in the VIME paper as closely as possible.

![image](https://github.com/Alcoholrithm/VIME---pytorch-implementation/assets/29500858/418e3167-24e1-4c61-b4bb-baa1ff2c652c)

<img width="1172" alt="image" src="https://github.com/Alcoholrithm/VIME---pytorch-implementation/assets/29500858/8fd1a43b-9ef3-4978-aa5a-65f912b7e88e">

# Install

```
pip install -r requirements.txt
```

# Usage

```
See the example_*.ipynb
```

# Difference between [the official tensorflow implementation](https://github.com/jsyoon0823/VIME)

The official implementation generates static mask vectors, feature vectors, and corrupted samples during initialization of the dataset. 

However, we generate them dynamically whenever call '\_\_getitem\_\_' of the dataset to avoid bias.
