# Text-to-Image-Synthesis 

## Intoduction

This is a pytorch implementation of [Generative Adversarial Text-to-Image Synthesis paper](https://arxiv.org/abs/1605.05396), forked from https://github.com/aelnouby/Text-to-Image-Synthesis. To this implementation we have added XLNet based embeddings and ESPCN super resolution.
<figure><img src='images/Pipeline.png'></figure>

## Requirements
All requirements have been added to requirements.txt. All requirements can be automatically installed via `pip install -r requirements.txt`
This implementation currently only support running with GPUs.

## Implementation details

This implementation follows the base https://github.com/aelnouby/Text-to-Image-Synthesis
We have added XLNet based text embeddings into the pipeline as well as ESPCN based super resolution model


## Datasets

We used [Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset, we converted each dataset (images, text embeddings) to hd5 format. 

**To use this code you can either:**

- Use the converted hd5 datasets,  [birds](https://drive.google.com/open?id=1mNhn6MYpBb-JwE86GC1kk0VJsYj-Pn5j), [flowers](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8)
- Convert the data youself
  1. download the dataset as described [here](https://github.com/reedscot/cvpr2016)
  2. Add the paths to the dataset to `config.yaml` file.
  3. Use [convert_cub_to_hd5_script](convert_cub_to_hd5_script.py) or [convert_flowers_to_hd5_script](convert_flowers_to_hd5_script.py) script to convert the dataset.
  
**Hd5 file taxonomy**
`
 - split (train | valid | test )
    - example_name
      - 'name'
      - 'img'
      - 'embeddings'
      - 'class'
      - 'txt'
      
## Usage
It would be recommeded to create a virtual environment before running the project
```
git clone https://github.com/anandshridhar/Text-to-Image-Synthesis.git
cd Text-to-Image-Synthesis
pip install -r requirements.txt
python -m visdom.server&
```
### Training

`python runtime.py

**Arguments:**
- `type` : GAN archiecture to use `(gan | wgan | vanilla_gan | vanilla_wgan)`. default = `gan`. Vanilla mean not conditional
- `dataset`: Dataset to use `(birds | flowers)`. default = `flowers`
- `split` : An integer indicating which split to use `(0 : train | 1: valid | 2: test)`. default = `0`
- `lr` : The learning rate. default = `0.0002`
- `diter` :  Only for WGAN, number of iteration for discriminator for each iteration of the generator. default = `5`
- `vis_screen` : The visdom env name for visualization. default = `gan`
- `save_path` : Path for saving the models.
- `l1_coef` : L1 loss coefficient in the generator loss fucntion for gan and vanilla_gan. default=`50`
- `l2_coef` : Feature matching coefficient in the generator loss fucntion for gan and vanilla_gan. default=`100`
- `pre_trained_disc` : Discriminator pre-tranined model path used for intializing training.
- `pre_trained_gen` Generator pre-tranined model path used for intializing training.
- `batch_size`: Batch size. default= `64`
- `num_workers`: Number of dataloader workers used for fetching data. default = `8`
- `epochs` : Number of training epochs. default=`200`
- `cls`: Boolean flag to whether train with cls algorithms or not. default=`False`


## Results

### Generated Images

<p align='center'>
<img src='images/64_flowers.jpeg'>
</p>

## Text to image synthesis
| Text        | Generated Images  |
| ------------- | -----:|
| A blood colored pistil collects together with a group of long yellow stamens around the outside        | <img src='images/examples/a blood colored pistil collects together with a group of long yellow stamens around the outside whic.jpg'>  |
| The petals of the flower are narrow and extremely pointy, and consist of shades of yellow, blue      | <img src='images/examples/the petals of the flower are narrow and extremely pointy, and consist of shades of yellow, blue and .jpg'>  |
| This pale peach flower has a double row of long thin petals with a large brown center and coarse loo | <img src='images/examples/this pale peach flower has a double row of long thin petals with a large brown center and coarse loo.jpg'> |
| The flower is pink with petals that are soft, and separately arranged around the stamens that has pi | <img src='images/examples/the flower is pink with petals that are soft, and separately arranged around the stamens that has pi.jpg'> |
| A one petal flower that is white with a cluster of yellow anther filaments in the center | <img src='images/examples/a one petal flower that is white with a cluster of yellow anther filaments in the center.jpg'> |


## References
[1]  Generative Adversarial Text-to-Image Synthesis https://arxiv.org/abs/1605.05396

[2]  5.	Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding. arXiv preprint arXiv:[1906.08237] XLNet: Generalized Autoregressive Pretraining for Language Understanding (arxiv.org)

[3]  7.	Shi, W., et al. "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." Papers with Code, https://paperswithcode.com/paper/real-time-single-image-and-video-super.


