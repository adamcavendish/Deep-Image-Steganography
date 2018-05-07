# StegNet Paper: Mega-Image-Steganography-Capacity-with-Deep-Convolutional-Network
---

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

