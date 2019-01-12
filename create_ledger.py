import os
from plumbum import cli
import pandas as pd
from collections import OrderedDict
from datetime import date

IMAGE_EXTENSIONS = ['jpg','jpeg', 'bmp', 'png', 'gif']
DETECTION_CLASSES = ["fire"]
TRAIN_CLASSES = ["background"] + DETECTION_CLASSES

class CreateLedger(cli.Application):
    DESCRIPTION = "Create a ledger"

    def main(self, directory):
        source = os.path.split(os.path.dirname(directory))[-1]
        print("Creating ledger for '{}':".format(source))
        data = []
        image_directory = os.path.join(directory, 'images')
        # Get subdirectories as labels
        labels = next(os.walk(image_directory))[1]
        for label in labels:
            image_filenames = [
                fn for fn in os.listdir(os.path.join(image_directory, label))
                if any(fn.endswith(ext) for ext in IMAGE_EXTENSIONS)
            ]

            for image_filename in image_filenames:
                item = {
                    "image": os.path.join(image_directory, label, image_filename),
                    "label": label,
                    "label_index": TRAIN_CLASSES.index(label),
                    "source": source,
                }
                data.append(item)


        df = pd.DataFrame(data)
        for col in list(df['label'].unique()):
            print("  * Found {} '{}' images".format(len(df[df['label'] == col]), col))

        ledger_pkl_filepath = os.path.join(directory, "ledger.pkl")
        df.to_pickle(ledger_pkl_filepath)
        print("  - Ledger saved to {}".format(ledger_pkl_filepath))

if __name__ == "__main__":
    CreateLedger.run()