"""Local YOLO inputs: no network, real training, or dataset edits."""
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from fusion.data import inspect_data, convert_data, read_data


def yolo_fixture(root, splits=("train", "val", "test"), image_dir="images"):
    config = {"nc": 2, "names": ["UGV", "USV"], "test": "../test/" + image_dir,
              "roboflow": {"workspace": "fixture", "project": "vehicles", "version": 3}}
    for split in splits:
        folder = "valid" if split == "val" else split
        images, labels = root / folder / image_dir, root / folder / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        config[split] = f"../{folder}/{image_dir}"
        for i in range(2):
            name = f"image {i}"
            color = (40 + i, 40 + 20 * ("train", "val", "test").index(split), 90)
            Image.new("RGB", (64, 64), color).save(images / (name + ".png"))
            (labels / (name + ".txt")).write_text("0 .4 .5 .3 .3\n1 .7 .5 .2 .2\n" if i == 0 else "", encoding="utf-8")
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class YoloData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="yolo data ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.http = patch("aiohttp.ClientSession._request", side_effect=AssertionError("Network forbidden"))
        self.http.start()
        self.addCleanup(self.http.stop)

    def test_negative_images_and_both_directory_names(self):
        for image_dir in ("images", "imgs"):
            with self.subTest(image_dir=image_dir):
                path = yolo_fixture(self.root / image_dir, splits=("train", "val"), image_dir=image_dir)
                before = {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
                info = inspect_data(path)
                self.assertEqual(info["names"], {0: "UGV", 1: "USV"})
                self.assertEqual(info["counts"]["train"], {"images": 2, "boxes": 2, "boxless_images": 1})
                self.assertEqual(info["counts"]["test"]["images"], 0)
                self.assertEqual(info["source"]["version"], 3)
                prepared = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
                config = yaml.safe_load(prepared.read_text())
                self.assertEqual(config["expected_images"]["val"], 2)
                self.assertNotIn("test", config)
                self.assertEqual(len(list(prepared.parent.glob("images/*/*"))), 4)
                labels = list(prepared.parent.glob("labels/*/*.txt"))
                self.assertEqual(sum(not p.read_text().strip() for p in labels), 2)
                self.assertEqual(before, {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()})
                self.assertEqual(prepared, asyncio.run(convert_data(path, cache_root=self.root / "cache")))

    def test_missing_labels_and_invalid_annotations(self):
        path = yolo_fixture(self.root / "invalid")
        label = path.parent / "train/labels/image 0.txt"
        for text in ("8 .5 .5 .2 .2", "0 .99 .5 .3 .3", "0 nan .5 .2 .2", "0 .5 .5 .2", "0.5 .5 .5 .2 .2"):
            with self.subTest(text=text):
                label.write_text(text)
                with self.assertRaises(ValueError):
                    inspect_data(path)
        label.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "label"):
            inspect_data(path)

    def test_missing_test_and_test_only_yml(self):
        path = yolo_fixture(self.root / "no test", splits=("train", "val"))
        with self.assertRaisesRegex(ValueError, "test"):
            inspect_data(path, required=("test",))
        path = yolo_fixture(self.root / "test only", splits=("test",)).rename(self.root / "test only/data.yml")
        self.assertEqual(inspect_data(path, required=("test",))["counts"]["test"]["images"], 2)
        prepared = asyncio.run(convert_data(path, required=("test",), cache_root=self.root / "cache"))
        self.assertEqual(yaml.safe_load(prepared.read_text())["expected_images"]["test"], 2)

    def test_content_changes_invalidate_cache_and_moves_do_not(self):
        path = yolo_fixture(self.root / "original")
        first = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        shutil.copytree(path.parent, self.root / "moved")
        self.assertEqual(inspect_data(path)["sha256"], inspect_data(self.root / "moved/data.yaml")["sha256"])
        (path.parent / "train/labels/image 0.txt").write_text("0 .5 .5 .2 .2\n")
        second = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        self.assertNotEqual(first, second)
        Image.new("RGB", (64, 64), (1, 2, 3)).save(path.parent / "train/images/image 0.png")
        third = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        self.assertNotEqual(second, third)
        config = yaml.safe_load(path.read_text())
        config["names"] = ["USV", "UGV"]
        path.write_text(yaml.safe_dump(config))
        self.assertNotEqual(third.parent.name, inspect_data(path)["sha256"])

    def test_cross_split_duplicate_and_corrupt_image(self):
        path = yolo_fixture(self.root / "duplicate")
        image = path.parent / "valid/images/image 1.png"
        original = image.read_bytes()
        shutil.copyfile(path.parent / "train/images/image 1.png", image)
        with self.assertRaisesRegex(ValueError, "across splits"):
            inspect_data(path)
        image.write_bytes(original)
        image.write_bytes(b"invalid image")
        with self.assertRaises(OSError):
            inspect_data(path)

    def test_configuration_and_path_boundaries(self):
        path = yolo_fixture(self.root / "config")
        config = yaml.safe_load(path.read_text())
        for updates in ({"nc": 3}, {"names": {1: "UGV", 2: "USV"}}, {"train": "../../outside/images"},
                        {"val": "missing/images"}, {"train": ["train/images"]}):
            with self.subTest(updates=updates):
                path.write_text(yaml.safe_dump({**config, **updates}))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    inspect_data(path)
        path.write_text(yaml.safe_dump({**config, "names": {0: "UGV", 1: "USV"}, "train": "train/images"}))
        self.assertEqual(inspect_data(path)["names"], {0: "UGV", 1: "USV"})

    def test_cached_image_tampering_is_rejected(self):
        path = yolo_fixture(self.root / "cache source")
        prepared = asyncio.run(convert_data(path, cache_root=self.root / "cache"))
        image = next(prepared.parent.glob("images/train/*"))
        Image.new("RGB", (64, 64), (4, 5, 6)).save(image)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            asyncio.run(convert_data(path, cache_root=self.root / "cache"))

    def test_ndjson_metadata_cannot_override_local_image_root(self):
        path = yolo_fixture(self.root / "legacy")
        rows, _ = read_data(path)
        rows[0]["local_root"] = str(self.root / "unrelated")
        legacy = path.with_suffix(".ndjson")
        legacy.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        prepared = asyncio.run(convert_data(legacy, cache_root=self.root / "cache"))
        self.assertEqual(len(list(prepared.parent.glob("images/*/*"))), 6)


if __name__ == "__main__":
    unittest.main()
