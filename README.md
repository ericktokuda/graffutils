# graffiti-deeplab
CV tasks using graffiti data


## MM segmentation

We need to install mmcv

```
conda create --name mmcv -y python=3.7 pip
conda deactivate && conda activate mmcv
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmcv-full=1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
cd /tmp/
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.19.0
pip install -e .
pip install pandas ipython erikunicamp-myutils
```

To run, copy the created folder to `~/projects`:
```
cd ~/projects/mmsegmentation/
mkdir checkpoints/
wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth -P checkpoints/
python ~/projects/graffutils/src/mmseg_ex_train.py
```

