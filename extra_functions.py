import csv
import os
from pathlib import Path


def output_to_CSV(data, name=None, specialColumns=False, labels=None):
    if labels is None:
        labels = []

    if name is None:
        output = data + "_rating.csv"
    else:
        output = name + ".csv"

    tempPath = Path(output)
    if tempPath.is_file():
        os.remove(output)
        print("\nDeleted: ", output)

    with open(output, 'w', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n')
        if labels:
            writer.writerow([label for label in labels])

        if not specialColumns:
            for key, value in data.items():
                writer.writerow([key, str(value)])
        else:
            for key, values in data.items():
                counter = 0
                for i in range(len(values)):
                    if counter == 0:
                        writer.writerow([key, values[0], values[1]])
                        counter += 1
                    else:
                        continue