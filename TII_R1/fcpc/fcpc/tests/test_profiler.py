from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.utils.profiler import ResourceMonitor, state_dict_nbytes


class ResourceMonitorTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_state_dict_payload_bytes(self) -> None:
        state = {
            "float": torch.zeros(3, dtype=torch.float32),
            "integer": torch.zeros(2, dtype=torch.int64),
        }
        self.assertEqual(state_dict_nbytes(state), 3 * 4 + 2 * 8)

    def test_cpu_monitor_returns_nonnegative_values(self) -> None:
        monitor = ResourceMonitor(device="cpu", interval_s=0.02).start()
        sum(i * i for i in range(10000))
        stats = monitor.stop()
        self.assertGreaterEqual(stats.process_cpu_mean_pct, 0.0)
        self.assertGreaterEqual(stats.rss_peak_mib, 0.0)


if __name__ == "__main__":
    unittest.main()
