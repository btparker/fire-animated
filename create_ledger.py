import os
from plumbum import cli
import pandas as pd
from collections import OrderedDict
from datetime import date

IMAGE_EXTENSIONS = ['jpg','jpeg', 'bmp', 'png', 'gif']

class CreateLedger(cli.Application):
    DESCRIPTION = "Create a ledger"

    def main(self, directory):
        source = os.path.split(os.path.dirname(directory))[-1]
        print("Creating ledger for '{}':".format(source))
        data = []
        # Get subdirectories, names are the tags to save
        subdirs = next(os.walk(directory))[1]
        for subdir in subdirs:
            image_filenames = [
                fn for fn in os.listdir(os.path.join(directory, subdir))
                if any(fn.endswith(ext) for ext in IMAGE_EXTENSIONS)
            ]

            for image_filename in image_filenames:
                item = {
                    "image": os.path.join(subdir, image_filename),
                    "category": subdir,
                    "source": source,
                }
                data.append(item)


        df = pd.DataFrame(data)
        for col in list(df['category'].unique()):
            print("  * Found {} '{}' images".format(len(df[df['category'] == col]), col))

        ledger_pkl_filepath = os.path.join(directory, "ledger.pkl")
        df.to_pickle(ledger_pkl_filepath)
        print("  - Ledger saved to {}".format(ledger_pkl_filepath))

        ledger_csv_filepath = os.path.join(directory, "ledger.csv")
        df.to_csv(ledger_csv_filepath)
        print("  - Ledger saved to {}".format(ledger_csv_filepath))


if __name__ == "__main__":
    CreateLedger.run()