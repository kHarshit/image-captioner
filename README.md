# image-captioner

Image Captioning using Encoder-Decoder

https://imagecaptioner.herokuapp.com/

`</>` **WORK IN PROGRESS**

## Overview

Recurrent Neural Networks (RNN) are used for varied number of applications including machine translation. The Encoder-Decoder architecture is utilized for such settings where a varied-length input sequence is mapped to the varied-length output sequence. The same network can also be used for image captioning.


In image captioning, the core idea is to use CNN as encoder and a normal RNN as decoder. This application uses the architecture proposed by [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555).

![image captioner structure](pytorch/image_captioner_structure.png)

Here's an excerpt from the paper:
> Here, we propose to follow this elegant recipe, replacing the encoder RNN by a deep convolution neural network (CNN).  Over  the  last  few  years  it  has  been  convincingly shown that CNNs can produce a rich representation of the input image by embedding it to a fixed-length vector, such that this representation can be used for a variety of vision tasks. Hence, it is natural to use a CNN as an image “encoder”, by first pre-training it for an image classification task and using the last hidden layer as an input to the RNN decoder that generates sentences. We call this model the **Neural Image Caption**, or **NIC**.

## Implementation

_All the code related to model implementation is in the [pytorch](pytorch) directory._

* **Dataset used:** MS-COCO dataset
* **Encoder:** The ResNet101 model pretrained on Imagenet is used as encoder.
* **Decoder:** The LSTM (Long-Short Term Memory) network is used as decoder.
