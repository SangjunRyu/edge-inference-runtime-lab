"""Checks for manifest order, input parity, finite sources and damaged files."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench.runtime.pipeline import preprocess
from bench.runtime.sources import make_source


class ImageListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        entries = []
        for i, shape in enumerate([(40, 60, 3), (80, 32, 3), (33, 49, 3)]):
            frame = np.random.default_rng(i).integers(0, 256, shape, dtype=np.uint8)
            data = cv2.imencode('.jpg', frame)[1].tobytes()
            name = f'{i}.jpg'
            (self.root / name).write_bytes(data)
            entries.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest()})
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'schema': 'image-classification-list/v1',
                                            'images': entries[::-1]}))

    def tearDown(self):
        self.temp.cleanup()

    def test_same_order_and_preprocessing_without_mutation(self):
        files = make_source(str(self.manifest), 3, 'files')
        memory = make_source(str(self.manifest), 3, 'predecoded')
        self.assertEqual(files.describe()['input_order_sha256'],
                         memory.describe()['input_order_sha256'])
        for disk, ram in zip(files.frames(), memory.frames()):
            np.testing.assert_array_equal(disk, ram)
            self.assertFalse(ram.flags.writeable)
            original = ram.copy()
            np.testing.assert_array_equal(preprocess(disk, 224, 224),
                                          preprocess(ram, 224, 224))
            np.testing.assert_array_equal(original, ram)
        self.assertIs(next(memory.frames()), memory.decoded[0])
        self.assertEqual(len(list(memory.frames())), 3)

    def test_no_silent_repeat_or_damaged_input(self):
        with self.assertRaises(ValueError):
            make_source(str(self.manifest), 4)
        (self.root / '0.jpg').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            make_source(str(self.manifest), 3)

    def test_reject_escape_from_manifest_directory(self):
        obj = json.loads(self.manifest.read_text())
        obj['images'][0]['path'] = '../outside.jpg'
        self.manifest.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, 'outside'):
            make_source(str(self.manifest), 1)


if __name__ == '__main__':
    unittest.main()
