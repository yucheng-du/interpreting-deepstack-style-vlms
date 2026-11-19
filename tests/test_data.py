import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepstack.data import OFFICIAL_DISCLAIMER, canonical_answer, official_prompt, option_labels, sample_indices, validate_record
from scripts.prepare_blink import download, partition_records


class BlinkDataTests(unittest.TestCase):
    def record(self):
        raw = "Which?\n(A) first\n(B) second"
        return {"id": "val_Art_Style_3", "task": "Art_Style", "choices": ["first", "second"],
                "option_labels": ["(A)", "(B)"], "answer_label": "(B)", "answer_index": 1,
                "raw_prompt": raw, "prompt": official_prompt("Art_Style", raw),
                "source": {"split": "val"}, "images": ["/external/first.png", "/external/second.png"],
                "image_metadata": [{"source_field": field, "width": 100, "height": 100}
                                   for field in ("image_1", "image_2")]}

    def test_actual_choice_set_and_official_disclaimer(self):
        self.assertEqual(option_labels(["one", "two"]), ["(A)", "(B)"])
        self.assertEqual(len(option_labels([str(i) for i in range(5)])), 5)
        self.assertEqual(official_prompt("Art_Style", "prompt"), OFFICIAL_DISCLAIMER + "prompt")
        self.assertEqual(official_prompt("Counting", "prompt"), "prompt")
        validate_record(self.record(), check_files=False)

    def test_label_image_and_prompt_corruption_rejected(self):
        mutations = [lambda r: r.update(answer_label="(C)"),
                     lambda r: r.update(answer_index=0),
                     lambda r: r["image_metadata"].reverse(),
                     lambda r: r.update(prompt=r["raw_prompt"]),
                     lambda r: r.update(choices=["second", "first"])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = copy.deepcopy(self.record())
                mutate(record)
                with self.assertRaises(ValueError):
                    validate_record(record, check_files=False)

    def test_sampling_nested_task_local_and_without_replacement(self):
        small = sample_indices(117, 8, 20260926, "Art_Style")
        big = sample_indices(117, 16, 20260926, "Art_Style")
        self.assertEqual(small, big[:8])
        self.assertEqual(small, sample_indices(117, 8, 20260926, "Art_Style"))
        self.assertEqual(len(set(small)), 8)
        self.assertNotEqual(small, sample_indices(117, 8, 20260926, "Counting"))
        full = sample_indices(117, 117, 20260926, "Art_Style")
        self.assertEqual(full[:8], small)
        self.assertEqual(sorted(full), list(range(117)))

    def test_full_partition_preserves_records_and_rejects_drift(self):
        full = [{"id": "a", "prompt": "first"}, {"id": "b", "prompt": "second"},
                {"id": "c", "prompt": "third"}]
        before = copy.deepcopy(full)
        previous = [copy.deepcopy(full[2]), copy.deepcopy(full[0])]
        old, new = partition_records(full, previous)
        self.assertEqual(old, [full[0], full[2]])
        self.assertEqual(new, [full[1]])
        self.assertEqual(full, before)
        for bad in ([{"id": "missing"}], [full[0], full[0]],
                    [{"id": "a", "prompt": "changed"}]):
            with self.subTest(previous=bad), self.assertRaises(ValueError):
                partition_records(full, bad)
        with self.assertRaises(ValueError):
            partition_records(full + [full[0]], previous)

    def test_offline_preparation_never_downloads_and_checks_cached_bytes(self):
        content = b"cached source"
        checksum = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen") as open_url:
            path = Path(directory) / "source.parquet"
            with self.assertRaises(FileNotFoundError):
                download(path.resolve().as_uri(), path, checksum, len(content), offline=True)
            path.write_bytes(content)
            self.assertEqual(download("unused", path, checksum, len(content), offline=True), checksum)
            with self.assertRaises(ValueError):
                download("unused", path, checksum, len(content) + 1, offline=True)
            with self.assertRaises(ValueError):
                download("unused", path, "0" * 64, len(content), offline=True)
            open_url.assert_not_called()

    def test_source_choice_capitalization_may_differ_from_prompt(self):
        record = self.record()
        record["choices"] = ["First", "Second"]
        validate_record(record, check_files=False)

    def test_bare_source_labels_normalized_without_remapping(self):
        labels = ["(A)", "(B)", "(C)"]
        self.assertEqual(canonical_answer("C", labels), "(C)")
        self.assertEqual(canonical_answer("(A)", labels), "(A)")
        with self.assertRaises(ValueError):
            canonical_answer("D", labels)


if __name__ == "__main__":
    unittest.main()
