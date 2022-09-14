# CycleGAN and Pix2Pix
This image contains the implementation of Pix2Pix and CycleGANs from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

It has been modified to fix the GPU CUDA drivers not working on the original image.

## Build the image
```sh
gradlew buildImage
```
## Create a new container with GPU cappabilities
The folowing command:
- Open the port 8097 to monitor the training.
- Enable the use of the GPU from the image.
- Allocates a pseudo-tty with STDIN
```sh
docker run --gpus all -p 8097:8097 -v <dataset_path>:/workspace/pytorch-CycleGAN-and-pix2pix/datasets/<dataset_name> -it  poc/pytorch-cyclegan-and-pix2pix:0.1.0 /bin/bash
```
## CycleGANs usage:
### Train
```sh
python train.py --dataroot ./datasets/<dataset_name> --name <model_name> --model cycle_gan
```
### Test
Once the model was trained, it can be tested executing:
```sh
python test.py --dataroot ./datasets/<dataset_name>  --name <model_name> --model cycle_gan
```
Results will be placed in: /workspace/pytorch-CycleGAN-and-pix2pix/results/<model_name>/test_latest

## Pix2Pix usage:
### Dataset preparation
Paired images needs to be joined together before using it in the algorithm.
Images to be joined on folders trainA and trainB shalll have the same name. The same applies to testA and testB folders.
Execute the following script to join paired images.
```sh
python make_dataset_aligned.py --dataset-path /workspace/pytorch-CycleGAN-and-pix2pix/datasets/<dataset_name>
```

### Train
Once the images are paired and joined execute:
```sh
python train.py --dataroot ./datasets/<dataset_name> --name <model_name> --model pix2pix --direction <AtoB|BtoA>
```

### Test
Once the model was trained, it can be tested executing:
```sh
python test.py --dataroot ./datasets/<dataset_name>  --name <model_name> --model pix2pix --direction <AtoB|BtoA>
```
Results will be placed in: /workspace/pytorch-CycleGAN-and-pix2pix/results/<model_name>/test_latest


## Satellite

docker run --gpus all --shm-size=8g -p 8097:8097 -v ${PWD}/datasets/satellital:/workspace/pytorch-CycleGAN-and-pix2pix/datasets/satellital -it pix2pix-tfm:latest /bin/bash

python -m visdom.server

python train.py --dataroot ./datasets/satellital --name satellital --model satellite --direction AtoB --dataset_mode satellite --input_nc 5 --output_nc 13 --n_epochs 150 --n_epochs_decay 100 --netG unet_256 --gan_mode lsgan --serial_batches

python test.py --dataroot ./datasets/satellital --name satellital --model satellite --direction AtoB --dataset_mode satellite --input_nc 2 --output_nc 13 --netG unet_256
