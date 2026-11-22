import unittest

import numpy as np

from bench.classification.validate_image_runtime import Ingress, InputPool


class IngressTest(unittest.TestCase):
    def test_full_queue_keeps_first_or_latest_ids(self):
        for policy, kept, dropped in (
            ("drop-oldest", [1, 2], [0]),
            ("drop-newest", [0, 1], [2]),
        ):
            with self.subTest(policy=policy):
                ingress = Ingress(2, policy)
                for image_id in range(3):
                    ingress.offer(image_id, 0)
                ingress.close()
                self.assertEqual([ingress.take()[0] for _ in range(2)], kept)
                self.assertIsNone(ingress.take())
                self.assertEqual(ingress.dropped, dropped)
                self.assertEqual(ingress.peak, 2)


class InputPoolTest(unittest.TestCase):
    def test_slot_is_reusable_after_callback_release(self):
        pool = InputPool(1, (1, 3, 2, 2), audit=True)
        tensor = np.ones((1, 3, 2, 2), dtype=np.float32)
        first, second = {"idx": 0}, {"idx": 1}
        original = pool.stage(first, tensor)
        pool.release(first)
        self.assertIs(pool.stage(second, tensor), original)
        with self.assertRaisesRegex(RuntimeError, "non-owner"):
            pool.release(first)
        pool.release(second)
        self.assertEqual(pool.stats()["outstanding"], 0)
        self.assertEqual(pool.stats()["integrity_failures"], 0)

    def test_fingerprint_detects_inflight_mutation(self):
        pool = InputPool(1, (1, 3, 2, 2), audit=True)
        rec = {"idx": 0}
        slot = pool.stage(rec, np.zeros((1, 3, 2, 2), dtype=np.float32))
        slot[0, 0, 0, 0] = 1
        pool.release(rec)
        self.assertEqual(pool.stats()["integrity_failures"], 1)
