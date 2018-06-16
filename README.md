# StegNet Paper: Mega-Image-Steganography-Capacity-with-Deep-Convolutional-Network
---

## Paper

[PDF Version](./paper/StegNet-Mega-Image-Steganography-Capacity-with-Deep-Convolutional-Network.pdf)

[HTML Version](http://www.mdpi.com/1999-5903/10/6/54/htm)


## How to create ImageNet Dataset used by StegNet

[Read the LMDB Creator Doc](./lmdb_creator/README.md)


## How to run the StegNet Model

Step 1. Setup Environmental Variables:

```bash
export ILSVRC2012_MDB_PATH="<Your Path to Created 'ILSVRC2012_image_train.mdb' Directory>"
```

Step 2. Run the code

```bash
python ./main.py
```

The command line arguments can be tweeked:
```
  -h, --help
  --train_max_epoch TRAIN_MAX_EPOCH
  --batch_size BATCH_SIZE
  --restart                          # Restart from scratch
  --global_mode {train,inference}
```

## Please cite as

```
@Article{StegNet,
  AUTHOR = {Wu, Pin and Yang, Yang and Li, Xiaoqiang},
  TITLE = {StegNet: Mega Image Steganography Capacity with Deep Convolutional Network},
  JOURNAL = {Future Internet},
  VOLUME = {10},
  YEAR = {2018},
  NUMBER = {6},
  ARTICLE NUMBER = {54},
  URL = {http://www.mdpi.com/1999-5903/10/6/54},
  ISSN = {1999-5903},
  DOI = {10.3390/fi10060054}
}
```
