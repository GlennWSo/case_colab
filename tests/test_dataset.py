# helpers
from pytest import fixture
import numpy as np
import os
from lungdata.utils import str2slice

# test target
from lungdata.dataset import DataSet

data_path = "tests/test.pickle"


def mk_data(s: slice):
    DataSet.load_wavs(s=s).save_pickle(data_path)


if not os.path.exists(data_path):
    mk_data(slice(10))


def load_data():
    return DataSet.load_pickle(data_path)


@fixture(scope="module")
def dataset() -> DataSet:
    return load_data()


def test_padding(dataset: DataSet):
    assert (
        not dataset.is_time_homo()
    ), "test data must be time hetro for test padding capablites"
    padded = dataset.pad()
    assert (
        not dataset.is_time_homo()
    ), "self.pad default behavior should not update self inplace"
    assert padded.is_time_homo(), "padded dataSet time dimensions should be consistent"
    dataset.pad(inplace=True)
    assert dataset == padded


def test_map(dataset: DataSet):
    def stats(x: np.ndarray, axis=None, transpose=False):
        """
        replace axis with mean, std, variance
        """
        m = x.mean(axis)
        s = x.std(axis)
        v = s**2

        a = np.array((m, s, v))
        if transpose:
            return a.transpose()
        return a

    mdata = dataset.map(stats, axis=1, transpose=True)
    oldtdim = dataset.max_tdim()
    assert mdata.max_tdim() == 3
    assert oldtdim > 3, "pad should not mutate by default"
    dataset.map(stats, axis=1, transpose=True, inplace=True)
    assert (
        mdata.features["mel"][0] == dataset.features["mel"][0]
    ).all(), "map_inplace should have same effect except it is inplace"


if __name__ == "__main__":
    reply = input("Create test.data pickle, input slice:\n")
    s = str2slice(reply)
    mk_data(s)
