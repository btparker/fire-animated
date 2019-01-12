import os
from plumbum import cli
import pandas as pd
import numpy as np
import h5py
import cv2
from PIL import Image
from sklearn.utils import shuffle

DETECTION_CLASSES = ["fire"]
TRAIN_CLASSES = ["background"] + DETECTION_CLASSES

IMG_DIMENSIONS = (738, 960)
IMG_SHAPE = (*IMG_DIMENSIONS, 3)

PADDING_FILL_COLOR=(127, 127, 127)

def train_validate_test_split(df, train_percent, validate_percent, seed=42):
    df = shuffle(df, random_state=seed)
    train_end = int(train_percent * len(df))
    validate_end = train_end + int(validate_percent * len(df))
    train = df[:train_end]
    validate = df[train_end:validate_end]
    test = df[validate_end:]
    return train, validate, test

def process_img(img):

    # Convert to PIL format
    img = resize_contain_img(
        img=img,
        desired_dimensions=IMG_DIMENSIONS,
    )
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def resize_contain_img(img, desired_dimensions):
    orig_h, orig_w, _ = img.shape
    des_h, des_w = desired_dimensions

    ratio = max(des_h, des_w)/ float(max(orig_h, orig_w))

    new_size = tuple([int(x*ratio) for x in [orig_h, orig_w]])
    new_h, new_w = new_size

    img = cv2.resize(img, (new_w, new_h))

    delta_w = des_w - new_w
    delta_h = des_h - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PADDING_FILL_COLOR)
    return new_img

class CreateDataset(cli.Application):
    DESCRIPTION = "Create a dataset"

    train_percent = cli.SwitchAttr(
        ['--train'],
        argtype=float,
        help='Dataset parition percentage for training',
        default=0.6,
    )
    
    validate_percent = cli.SwitchAttr(
        ['--validate'],
        argtype=float,
        help='Dataset parition percentage for validation',
        default=0.2,
    )

    output = cli.SwitchAttr(
        ['--output'],
        argtype=str,
        help='Output directory',
        mandatory=True,
    )

    def create_split_dataset(self, split_type, df, hdf5_file):
        pkl_filepath = os.path.join(self.output, "{}.pkl".format(split_type))
        df.to_pickle(pkl_filepath)
        print("- {}, {} items, {}".format(
            split_type,
            len(df),
            pkl_filepath,
        ))
        shape = (len(df), *IMG_SHAPE)

        labels = df['label_index'].values
        image_paths = df['image'].values

        hdf5_file.create_dataset("{}_img".format(split_type), shape, np.int8)

        if split_type == 'train':
            hdf5_file.create_dataset("train_mean", IMG_SHAPE, np.float32)

        hdf5_file.create_dataset("{}_labels".format(split_type), (len(df),), np.int8)
        hdf5_file["{}_labels".format(split_type)][...] = labels

        if split_type == 'train':
            # a numpy array to save the mean of the images
            mean = np.zeros(IMG_SHAPE, np.float32)

        for i, image_path in enumerate(image_paths):
            # print how many images are saved every 1000 images
            if i % 100 == 0 and i > 1:
                print('{} data: {}/{}'.format(split_type, i, len(image_paths)))
            
            img = cv2.imread(image_path)
            img = process_img(img=img)

            # cv2 load images as BGR, convert it to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # add any image pre-processing here
            # save the image and calculate the mean so far
            hdf5_file["{}_img".format(split_type)][i, ...] = img[None]
            if split_type == 'train':
                mean += img / float(len(labels))
        if split_type == 'train':
            hdf5_file["train_mean"][...] = mean

    def main(self, *ledgers):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        dataframes = []

        for ledger in list(ledgers):
            dataframes.append(pd.read_pickle(ledger))

        df = pd.concat(dataframes)
        print("Creating dataset '{}', total {} items:".format(self.output, len(df)))
        train_df, validate_df, test_df = train_validate_test_split(
            df=df,
            train_percent=self.train_percent,
            validate_percent=self.validate_percent,
        )

        hdf5_path = os.path.join(self.output, "dataset.hdf5")
        hdf5_file = h5py.File(hdf5_path, mode='w')
        self.create_split_dataset(split_type='train', df=train_df, hdf5_file=hdf5_file)
        self.create_split_dataset(split_type='validate', df=validate_df, hdf5_file=hdf5_file)
        self.create_split_dataset(split_type='test', df=test_df, hdf5_file=hdf5_file)
        hdf5_file.close()
        print("...Dataset saved to '{}'".format(hdf5_path))

if __name__ == "__main__":
    CreateDataset.run()