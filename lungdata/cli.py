from lungdata.dataset import DataSet
from lungdata.augment import DEFUALT_AUGS
from lungdata.utils import str2slice


def make_dataset():
    str_slice = input("load records: [slice]\n")
    print(f"building data using records[{str_slice}]")
    data = DataSet.load_wavs(DEFUALT_AUGS, s=str2slice(str_slice))
    print(data)
    print("---\n")

    path = input("save: [path/no]\n").strip()
    low = path.lower()
    if low == "no" or low == "n" or low == "":
        print("skipping save")
        return data
    data.save_pickle("test.data")
    return data


if __name__ == "__main__":
    make_dataset()
    data = DataSet.load_pickle("test.data")
    print(data)
