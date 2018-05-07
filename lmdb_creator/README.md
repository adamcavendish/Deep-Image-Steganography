# Create lmdb files of ImageNet 2012 Contest

Step 1. Download and Extract ILSVRC2012 Dataset

Step 2. Make sure that you have `ILSVRC2012_devkit_t12` and `ILSVRC2012_img_train` directory

Step 3. Set up environment variables:

```bash
export IMAGE_DIR="<Your ILSVRC2012_img_train Directory Path>"
export DK_DIR="<Your ILSVRC2012_devkit_t12 Directory Path>"
export MDB_OUT_DIR="<Your Expected Directory for Generating LMDB File>" # Note: Reserve 60GB at least
```

Step 4. Run the python script

```bash
python ./images2lmdb.py
```
