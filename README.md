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
