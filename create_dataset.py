import os
from plumbum import cli
import pandas as pd
import numpy as np
import h5py
import cv2

DETECTION_CLASSES = ["fire"]
TRAIN_CLASSES = ["background"] + DETECTION_CLASSES

IMG_DIMENSIONS = (738, 960)
IMG_SHAPE = (*IMG_DIMENSIONS, 3)

def train_validate_test_split(df, train_percent, validate_percent, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test

class CreateDataset(cli.Application):
    DESCRIPTION = "Create a dataset"

    train_percent = cli.SwitchAttr(
        ['--train'],
        argtype=float,
        help='Path to desired input moshcap ledgers',
        default=0.6,
    )
    
    validate_percent = cli.SwitchAttr(
        ['--validate'],
        argtype=float,
        help='Path to desired input moshcap ledgers',
        default=0.2,
    )

    output = cli.SwitchAttr(
        ['--output'],
        argtype=str,
        help='Path to desired dataset output folder',
        mandatory=True,
    )

    def main(self, *ledgers):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        dataframes = []

        for ledger in list(ledgers):
            dataframes.append(pd.read_pickle(ledger))

        df = pd.concat(dataframes)

        train_df, validate_df, test_df = train_validate_test_split(
            df=df,
            train_percent=self.train_percent,
            validate_percent=self.validate_percent,
        )

        train_pkl_filepath = os.path.join(self.output, "train.pkl")
        validate_pkl_filepath = os.path.join(self.output, "validate.pkl")
        test_pkl_filepath = os.path.join(self.output, "test.pkl")

        train_df.to_pickle(train_pkl_filepath)
        validate_df.to_pickle(validate_pkl_filepath)
        test_df.to_pickle(test_pkl_filepath)

        print("Creating dataset '{}', total {} items:".format(self.output, len(df)))
        print("- Train ({}%) {} items, {}".format(
            round(self.train_percent * 100, 2),
            len(train_df),
            train_pkl_filepath,
        ))
        print("- Validate ({}%) {} items, {}".format(
            round(self.validate_percent * 100, 2),
            len(validate_df),
            validate_pkl_filepath,
        ))
        print("- Test ({}%) {} items, {}".format(
            round((1.0 - self.train_percent - self.validate_percent) * 100, 2),
            len(test_df),
            test_pkl_filepath,
        ))

        train_shape = (len(train_df), *IMG_SHAPE)
        validate_shape = (len(validate_df), *IMG_SHAPE)
        test_shape = (len(test_df), *IMG_SHAPE)

        hdf5_path = os.path.join(self.output, "dataset.hdf5")
        hdf5_file = h5py.File(hdf5_path, mode='w')

        train_labels = train_df['label_index'].values
        validate_labels = validate_df['label_index'].values
        test_labels = test_df['label_index'].values

        train_image_paths = train_df['image'].values
        validate_image_paths = validate_df['image'].values
        test_image_paths = test_df['image'].values

        hdf5_file.create_dataset("train_img", train_shape, np.int8)
        hdf5_file.create_dataset("validate_img", validate_shape, np.int8)
        hdf5_file.create_dataset("test_img", test_shape, np.int8)
        hdf5_file.create_dataset("train_mean", IMG_SHAPE, np.float32)
        hdf5_file.create_dataset("train_labels", (len(train_df),), np.int8)
        hdf5_file["train_labels"][...] = train_labels
        hdf5_file.create_dataset("validate_labels", (len(validate_df),), np.int8)
        hdf5_file["validate_labels"][...] = validate_labels
        hdf5_file.create_dataset("test_labels", (len(test_df),), np.int8)
        hdf5_file["test_labels"][...] = test_labels

        # a numpy array to save the mean of the images
        mean = np.zeros(IMG_SHAPE, np.float32)
        # loop over train addresses
        for i in range(len(train_image_paths)):
            # print how many images are saved every 1000 images
            if i % 100 == 0 and i > 1:
                print('Train data: {}/{}'.format(i, len(train_image_paths)))
            # read an image and resize
            # cv2 load images as BGR, convert it to RGB
            addr = train_image_paths[i]
            img = cv2.imread(addr)
            img = cv2.resize(img, IMG_DIMENSIONS[::-1], interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here
            # save the image and calculate the mean so far
            hdf5_file["train_img"][i, ...] = img[None]
            mean += img / float(len(train_labels))
        # loop over validation addresses
        for i in range(len(validate_image_paths)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Validation data: {}/{}'.format(i, len(validate_image_paths)))
            # read an image and resize
            # cv2 load images as BGR, convert it to RGB
            addr = validate_image_paths[i]
            img = cv2.imread(addr)
            img = cv2.resize(img, IMG_DIMENSIONS[::-1], interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here
            # save the image
            hdf5_file["validate_img"][i, ...] = img[None]
        # loop over test addresses
        for i in range(len(test_image_paths)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Test data: {}/{}'.format(i, len(test_image_paths)))
            # read an image and resize
            # cv2 load images as BGR, convert it to RGB
            addr = test_image_paths[i]
            img = cv2.imread(addr)
            img = cv2.resize(img, IMG_DIMENSIONS[::-1], interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here
            # save the image
            hdf5_file["test_img"][i, ...] = img[None]
        # save the mean and close the hdf5 file
        hdf5_file["train_mean"][...] = mean
        hdf5_file.close()
        print("...Dataset saved to '{}'".format(hdf5_path))

if __name__ == "__main__":
    CreateDataset.run()