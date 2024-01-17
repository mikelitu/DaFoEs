# DaFoEs

Implementation of the models presented in the paper "DaFoEs: Mixing Datasets Towards the generalization of vision-state deep-learning Force Estimation in Minimally Invasive Robotic Surgery" to appear on RA-L 2024.

xfun::embed_file(imgs/Experimental_setup.pdf)

## Setup the environment

The following code has been tested with *Python3.11*. Setup the *conda* environment to run the code.

```shell
conda create -n dafoes python=3.11
conda activate dafoes
pip3 install -r requirements.txt
```

## Models

The [*models*](models) folder contain multiple scripts to build the different models presented on our paper. The list of models and the different combinations for the encoder and decoder can be found on the table below:

|       **Decoder**       |     **Encoder**    | **Network name** |
|:-----------------------:|:------------------:|:----------------:|
| **MLP (Non-recurrent)** |      ResNet50      |        CNN       |
|                         | Vision Transformer |        ViT       |
|                         |   Fully connected  |        MLP       |
|   **LSTM (Recurrent)**  |      ResNet50      |       R-CNN      |
|                         | Vision Transformer |       R-ViT      |


The main model constructor can be found on [*models/force_estimator.py*](models/force_estimator.py). The class **ForceEstimator** can be initialized with the following variables:

* **architecture**: A string that contains one of the possible 3 encoders ("cnn", "vit" or "fc").
* **recurrency**: Boolean to add recurrency to the model
* **pretrained**: Boolean to load the weights of the pre-trained ResNet50 or ViT model from Pytorch.
* **att_type**: Add some attention blocks to the CNN encoder
* **state_size**: The size for the state vector


## Training

The scripts are adapted for the use on our two available dataset, however it can be adapted to any custom dataset of your preference. The [*train.py*](train.py) file contains an example for a simple training loop on how to train the force estimation model.

```python3
python3 train.py --dataset <your-dataset-name> --architecture <desire-decoder> --recurrency
```

To include your custom dataset update the content on [*datasets/vision_state_dataset.py*](datasets/vision_state_dataset.py).


## Testing

To test the model use the following command:

```python3
python3 test.py --dataset <your-dataset-name> --architecture <trained_architecture> 
```

## License and copyright

Please see the [LICENSE](LICENSE) file for details.

## Acknowledgement

Special thanks to Zhonge Chua for sharing their dataset from his article [[1]](#1) with us for this project

## Citation

If you find this code or this research useful, please consider citing our work:

```bibtex

```

## References

<a id="1">[1]</a>
Chua, Z., Jarc, A. M., & Okamura, A. M. (2021, May). Toward force estimation in robot-assisted surgery using deep learning with vision and robot state. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 12335-12341). IEEE.

