<div align="center">

<h1><a href="link to be decided">paper to be decided</a></h1>


</div>


# GSCC
This is the pytorch implementation of the paper: "GSCC: Graph-Structral Relationship Quantification For Remote Sensing Image Change Captioning". 

For more information, please see our published paper in [[IEEE](link to be decided)]  ***(Accepted by To be decided)***



# Installation
You can complete the environment installation using either of the two methods below:

## Docker Image
To avoid potential issues during environment setup, we recommend using the provided docker image file to create a container.
You can download the image file from [[Baidu Pan](https://pan.baidu.com/s/1rc7SSviRneh9Q9E2Y-WLlg) (code:aco5)] (code:to be done).

If you have successfully created a container from the image file, skip to the data preparation step.

## Environment
* OS: Linux
* Python >= 3.7 (The python version in our environment is 3.7.11)
* Torch == 1.10.0 (CUDA == 11.3)

### Dependencies Installation
1. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
   
2. Install java
    ```
    apt install openjdk-11-jdk
    ```

3. Install torch-geometric:

    For the detailed installation document, please refer to [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), 
    and you can find the wheels you need on [PyG Wheels](https://data.pyg.org/whl/).
    Before installing torch-geometric, make sure you have installed the required packages following the document above.
    
    After installing the needed packages (we recommend installing these packages from wheels), install torch-geometric as follows:
    ```
    pip install torch-geometric==2.0.4
    ```
    Note: ensure the version of PyG is correct (2.0.4), or you might encounter some errors during training.
    If you have trouble installing PyG, please raise an issue, and we will reply as soon as we can.

# Data Preparation
For a fair comparison, we completely follow [RSICCformer](https://github.com/Chen-Yang-Liu/RSICC) to preprocess the LEVIR-CC dataset.
You can also directly download the processed files from their links.



# Test with Our Checkpoints
You can download our model checkpoints——by [[Baidu Pan](https://pan.baidu.com/s/176CHoJuXjn3gOSxnyaJH7w) (code:phyb)]

After downloading the pretrained weights, you can put it in `./models_checkpoint/` and reproduce the results reported in our paper as follows:
```python
python eval.py --gpu 0 --tag BEST_LEVIR-CC --sig_w 50.0 --top_k 112 --mask_head 8
```


# Train

```python
python train.py --gpu 0 --tag LEVIR_2025 --sig_w 50.0 --top_k 112 --mask_head 8 --seed 1
```

# Evaluate
```python
python eval.py --gpu 0 --tag LEVIR_2025 --sig_w 50.0 --top_k 112 --mask_head 8
```
The training process is still unstable even with a fixed seed, so you might need to train the model a few times to get results close to those in the paper.

# Citation: 
```
to be done
```
# Reference:
Thanks to the following repository:
[RSICCformer](https://github.com/Chen-Yang-Liu/RSICC)




