# Image Completion

![](completion.gif)

The network has been trained on 80% of the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and the faces used for completion have been randomly sampled from the remaining 20%.

This repository aims to implement Raymond Yeh and Chen Chen *et al.*'s paper [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/pdf/1607.07539).
It is done with the Torch framework and is based on Soumith's DCGAN implementation https://github.com/soumith/dcgan.torch.

## Usage

1. Install `torch`: http://torch.ch/docs/getting-started.html

2. Choose a dataset and create a folder with its name (ex: `mkdir celebA`). Inside this folder create a folder `images` containing your images.  
*Note:* for the `celebA` dataset, run
```
DATA_ROOT=celebA th data/crop_celebA.lua
```

3. Train a DCGAN model in order to obtain a discriminator and a generator networks. I have already trained the model on the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put the corresponding networks into the `checkpoints` folder. If you want to train it again or use a different dataset run
```
DATA_ROOT=<dataset_folder> name=<whatever_name_you_want> th main.lua
```

4. Complete your images. You may want to choose another dataset to avoid completing images you used for training.
```
DATA_ROOT=<dataset_folder> name=<whatever_name_you_want> net=<prefix_of_net_in_checkpoints> th inpainting.lua
```
*Example:*
```
DATA_ROOT=celebA noise=normal net=celebA-normal name=inpainting-celebA display=2929 th inpainting.lua
```

## Display images in a browser

If you want, install the `display` package (`luarocks install display`) and run
```
th -ldisplay.start <PORT_NUMBER> 0.0.0.0
```
to launch a server on the port you chose. You can access it in your browser with the url http://localhost:PORT_NUMBER.

To train your network or for completion add the variable `display=<PORT_NUMBER>` to the list of options.

## Optional parameters

In your command line instructions you can specify several parameters (for example the display port number), here are some of them:
+ `noise` which can be either `uniform` or `normal` indicates the prior distribution from which the samples are generated
+ `batchSize` is the size of the batch used for training or the number of images to reconstruct
+ `name` is the name you want to use to save your networks or the generated images
+ `gpu` specifies if the computations are done on the GPU or not. Set it to 0 to use the CPU (not recommended, see below) and to n to use the nth GPU you have (1 is the default value)
+ `lr` is the learning rate
+ `loadSize` is the size to use to scale the images. 0 means no rescale

## About the GPU

It is highly recommended to do the computations on the GPU since it will be much faster. However I provided networks that can be used without gpu in the `checkpoints/` folder (the ones with "float" in their name).
