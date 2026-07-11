import pickle

import numpy as np

from nanotrain.data import BinTokenDataset, ShakespeareCharDataset


def test_shakespeare_char_dataset_get_batch(tmp_path) -> None:
    train = np.arange(128, dtype=np.uint16)
    val = np.arange(64, dtype=np.uint16)
    train.tofile(tmp_path / "train.bin")
    val.tofile(tmp_path / "val.bin")
    with (tmp_path / "meta.pkl").open("wb") as f:
        pickle.dump({"vocab_size": 128}, f)

    dataset = ShakespeareCharDataset(
        data_dir=tmp_path,
        block_size=16,
        batch_size=4,
        device="cpu",
    )
    x, y = dataset.get_batch("train")

    assert dataset.vocab_size_from_meta() == 128
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_bin_token_dataset_get_batch(tmp_path) -> None:
    train = np.arange(256, dtype=np.uint16)
    val = np.arange(128, dtype=np.uint16)
    train.tofile(tmp_path / "train.bin")
    val.tofile(tmp_path / "val.bin")

    dataset = BinTokenDataset(
        data_dir=tmp_path,
        block_size=32,
        batch_size=2,
        device="cpu",
    )
    x, y = dataset.get_batch("val")

    assert dataset.vocab_size_from_meta() is None
    assert x.shape == (2, 32)
    assert y.shape == (2, 32)
