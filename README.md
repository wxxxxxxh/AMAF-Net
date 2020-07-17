# [AIM2020] Adaptively Multi-gradients Auxiliary Feature Learning for Efficient Super-resolution

## Dependecies
- python = 3.6
- pytorch = 1.5.0
- torchvision = 0.6.0
- cudatoolkit = 10.1
- tqdm = 4.48.0
- easydict = 1.9
- scikit-image = 0.17.2
- opencv-contrib-python = 4.3.0.36
- pyyaml = 5.3.1
- pillow = 6.2.2
- scipy = 1.2.1

If you use Anaconda, you can run these commands to build a env that can run our code correctly:
```bash
#First step: build a env and enter it
conda create -n AMAF python=3.6
conda activate AMAF

#Second step: install pytorch and torchvision via conda
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

#Final step: install other dependecies via pip
pip install -r requirements.txt
```
Now, you have build a conda env for running code successfully!


## Datasets
Please put the testing data (such like `DIV2K_test_LR_bicubic/X4`) that you want to check results in `./Datasets`, now you will have:

```
|- ./Datasets/
|--- DIV2K_test_LR_bicubic/
|------ X4/
|--------- 0901x4.png
|--------- 0902x4.png
|--------- ...
```

## Testing
We provide the pretrained model of AMAF in `./ckp`, so you can test images directly. Run this commands to obatain the results. They can be seen in `./results`:

```bash
python test.py --config ./yaml/test.yaml
```

## Others
#### Paramaters
- Parameters used in our code can be seen in `./yaml/test.yaml`
#### Input and output
- If you use `cv2` to load an image, you need to transfer it to (R,G,B) format.
- You are NOT asked to normalize the input/output image to 0\~1 or -1\~1, keep them be 0\~255.
