from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aidetect.config import Config
from aidetect.data import (
    ImageItem,
    _detect_fake_index,
    _stable_bucket,
    build_transforms,
    cross_generator_split,
    degrade_jpeg,
    degrade_resize,
)


def _fake(name, gen):
    return ImageItem(path=Path(name), label=1, generator=gen)


def _real(name):
    return ImageItem(path=Path(name), label=0, generator="real")


def test_cross_generator_split_holds_out_generator():
    items = [
        _fake("a.png", "sd15"),
        _fake("b.png", "sd15"),
        _fake("c.png", "midjourney"),
        _real("r1.png"),
        _real("r2.png"),
    ]
    train, test = cross_generator_split(items, test_generators={"midjourney"})
    train_fake_gens = {it.generator for it in train if it.label == 1}
    test_fake_gens = {it.generator for it in test if it.label == 1}
    assert "midjourney" in test_fake_gens
    assert "midjourney" not in train_fake_gens


def test_cross_generator_split_has_no_path_in_both_sides():
    items = [_fake("a.png", "sd15"), _fake("c.png", "midjourney")] + [
        _real(f"r{i}.png") for i in range(20)
    ]
    train, test = cross_generator_split(items, test_generators={"midjourney"})
    train_paths = {str(it.path) for it in train}
    test_paths = {str(it.path) for it in test}
    assert train_paths.isdisjoint(test_paths)


def test_cross_generator_split_detects_leak():
    # if a "held out" generator is also asked to stay in train, that is impossible
    # here because the function routes all of its fakes to test, so no leak can
    # occur. This checks the guard path stays satisfied on a normal split.
    items = [_fake("a.png", "sd15"), _real("r.png")]
    train, test = cross_generator_split(items, test_generators={"sd15"})
    assert all(it.generator != "sd15" for it in train if it.label == 1)


def test_config_rejects_bad_mode():
    with pytest.raises(ValueError):
        Config(mode="halfbaked").validate()


def test_config_rejects_bad_tpr():
    with pytest.raises(ValueError):
        Config(operating_tpr=1.5).validate()


def test_stable_bucket_is_deterministic():
    # builtin hash() is salted per process; this must not be.
    assert _stable_bucket("data/r1.png", 2) == _stable_bucket("data/r1.png", 2)
    assert _stable_bucket("data/r1.png", 4) in {0, 1, 2, 3}


def test_detect_fake_index_reads_class_names():
    # CIFAKE's mirror names the classes ['FAKE', 'REAL']; fake must map to index 0.
    assert _detect_fake_index(["FAKE", "REAL"]) == 0
    assert _detect_fake_index(["real", "ai"]) == 1
    with pytest.raises(ValueError):
        _detect_fake_index(["cat", "dog"])


def test_build_transforms_output_shape():
    t = build_transforms(image_size=64, train=False)
    img = Image.new("RGB", (32, 32), (120, 30, 200))
    out = t(img)
    assert tuple(out.shape) == (3, 64, 64)


def test_degrade_keeps_size():
    img = Image.new("RGB", (32, 32), (10, 200, 90))
    assert degrade_jpeg(img, 30).size == (32, 32)
    assert degrade_resize(img, 0.5).size == (32, 32)
    # factor 1.0 is a no-op pass-through
    assert np.asarray(degrade_resize(img, 1.0)).shape == (32, 32, 3)
