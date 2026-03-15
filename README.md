## set up new pytorch env 
conda create -n clothesswap python=3.11 -y

conda activate clothesswap

# use cpu if no cuda isn't available
conda install pytorch torchvision cpuonly -c pytorch -y

pip install transformers diffusers accelerate Pillow numpy opencv-python
