'''
Generate ILSVRC2012 Dataset LMDB file
'''
import io
import os
import pathlib
import struct
import sys
import time

import PIL.Image
import lmdb
import msgpack

import scipy.io
import cytoolz as tz
import numpy as np

# Prepare
IMAGE_DIR = os.environ['IMAGE_DIR']
DK_DIR = os.environ['DK_DIR']
MDB_OUT_DIR = os.environ['MDB_OUT_DIR']

seed = 42
np.random.seed(seed)

lmdb_map_size = 50*1024*1024*1024
lmdb_txn_size = 500

# Setup PATHs
META_PATH = os.path.join(DK_DIR, 'data', 'meta.mat')
META_MP_PATH = os.path.join(MDB_OUT_DIR, 'meta.msgpack')
LMDB_PATH = os.path.join(MDB_OUT_DIR, 'ILSVRC2012_image_train.mdb')

# Generate meta.msgpack
meta = scipy.io.loadmat(META_PATH, squeeze_me=True)
synsets = meta['synsets']

meta_info = [{
    'ILSVRC2012_ID':    int(s['ILSVRC2012_ID']),
    'WNID':             str(s['WNID']),
    'words':            str(s['words']),
    'gloss':            str(s['gloss']),
    'wordnet_height':   int(s['wordnet_height']),
    'num_train_images': int(s['num_train_images'])
} for s in synsets]

meta_info_packed = msgpack.packb(meta_info, use_bin_type=True)

with open(META_MP_PATH, 'wb') as f:
    f.write(meta_info_packed)

# Generate LMDB
def make_context():
    return {
        'image_id': 0,
        'clock_beg': time.time(),
        'clock_end': time.time(),
    }


def process_image_one(txn, image_id, wordnet_id, label, image_abspath):
    '''
    txn: lmdb transaction object
    image_id: int
      The image id, increasing index
    wordnet_id: str
      The wordnet id, i.e. n07711569
    image_abspath: str
      The image's absolute path
    '''
    with PIL.Image.open(image_abspath) as im, io.BytesIO() as bio:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        rows, cols = im.size
        cnls = 3
        im.resize((256, 256))
        im.save(bio, format='webp')
        image_bytes = bio.getvalue()

    filename = os.path.basename(image_abspath).rstrip('.JPEG')

    info = {
        'wordnet_id': wordnet_id,
        'filename': filename,
        'image': image_bytes,
        'rows': rows,
        'cols': cols,
        'cnls': cnls,
        'label': label,
    }
    key = '{:08d}'.format(image_id).encode()
    txn.put(key, msgpack.packb(info, use_bin_type=True))


def imagenet_walk(wnid_meta_map, image_Dir):
    def get_category_image_abspaths(Path):
        return [str(f.absolute()) for f in Path.iterdir() if f.is_file()]

    def process_category_one(count, category_Path):
        wordnet_id = category_Path.name
        metainfo = wnid_meta_map[wordnet_id]
        words = metainfo['words']
        gloss = metainfo['gloss']
        label = metainfo['ILSVRC2012_ID']

        print('Process count=%d, label=%d, wordnet_id=%s' % (count, label, wordnet_id))
        print('  %s: %s' % (words, gloss))
        for image_abspath in get_category_image_abspaths(category_Path):
            yield {
                'label': label,
                'wordnet_id': wordnet_id,
                'image_abspath': image_abspath
            }

    categories = [d for d in image_Dir.iterdir() if d.is_dir()]

    image_files = [
        image_info
        for count, category_Path in enumerate(categories)
        for image_info in process_category_one(count, category_Path)
    ]
    return image_files


def process_images(ctx, lmdb_env, image_infos, image_total):
    image_id = ctx['image_id']

    with lmdb_env.begin(write=True) as txn:
        for image_info in image_infos:
            wordnet_id = image_info['wordnet_id']
            label = image_info['label']
            image_abspath = image_info['image_abspath']
            process_image_one(txn, image_id, wordnet_id, label, image_abspath)
            image_id = image_id + 1

    clock_beg = ctx['clock_beg']
    clock_end = time.time()

    elapse = clock_end - clock_beg
    elapse_h = int(elapse) // 60 // 60
    elapse_m = int(elapse) // 60 % 60
    elapse_s = int(elapse) % 60

    estmt = (image_total - image_id) / image_id * elapse
    estmt_h = int(estmt) // 60 // 60
    estmt_m = int(estmt) // 60 % 60
    estmt_s = int(estmt) % 60

    labels = [image_info['label'] for image_info in image_infos]
    print('ImageId: {:8d}/{:8d}, time: {:2d}h/{:2d}m/{:2d}s, remain: {:2d}h/{:2d}m/{:2d}s, Sample: {} ...'.format(
        image_id, image_total,
        elapse_h, elapse_m, elapse_s,
        estmt_h, estmt_m, estmt_s,
        str(labels)[:80]))

    ctx['image_id'] = image_id
    ctx['clock_end'] = clock_end


wnid_meta_map = { m['WNID']: m for m in meta_info }

image_train_env = lmdb.open(LMDB_PATH, map_size=lmdb_map_size)

image_infos = imagenet_walk(wnid_meta_map, pathlib.Path(IMAGE_DIR))
image_total = len(image_infos)
np.random.shuffle(image_infos)

ctx = make_context()
for image_infos_partial in tz.partition_all(lmdb_txn_size, image_infos):
    process_images(ctx, image_train_env, image_infos_partial, image_total)
