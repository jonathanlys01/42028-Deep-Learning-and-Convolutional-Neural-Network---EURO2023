# Setup
### Create venv 
Go to the root directory, and 

    python -m venv project_env

### venv requirements

    pip install -r requirements.txt
Note that the numpy version is not the latest

### facebook repos
if not present: git clone detectron and videopose3d repos

    git clone https://github.com/facebookresearch/VideoPose3D.git
    git clone https://github.com/facebookresearch/detectron2.git

### Detectron2 requirements

    cd detectron2
    pip install -e .
   This will also setup the configs for the models.
   Then, go back to the root directory

###  Checkpoint for Videopose3d

Download the pretrained model:

    bash download_checkpoint.sh
### Current
If you stick to the `main.py` file, create a folder named current, with 2 subfolder input and output. 

All good.

See the end of the `main.py`file to change the path
 
