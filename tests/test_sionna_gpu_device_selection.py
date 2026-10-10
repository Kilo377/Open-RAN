import unittest

from phy.sionna_engine import SionnaEngine


class _FakeExperimentalConfig:
    def __init__(self) -> None:
        self.memory_growth_calls = []

    def set_memory_growth(self, device, enabled) -> None:
        self.memory_growth_calls.append((device, enabled))


class _FakeConfig:
    def __init__(self, gpus) -> None:
        self._gpus = list(gpus)
        self.experimental = _FakeExperimentalConfig()
        self.logical_device_configurations = []

    def list_physical_devices(self, device_type: str):
        if device_type == "GPU":
            return list(self._gpus)
        return []

    def set_logical_device_configuration(self, device, configs) -> None:
        self.logical_device_configurations.append((device, list(configs)))

    class LogicalDeviceConfiguration:
        def __init__(self, memory_limit: int) -> None:
            self.memory_limit = memory_limit


class _FakeTf:
    def __init__(self, gpus) -> None:
        self.config = _FakeConfig(gpus)


class TestSionnaGpuDeviceSelection(unittest.TestCase):
    def test_gpu_device_selects_gpu_and_enables_memory_growth(self) -> None:
        engine = SionnaEngine({"device": "gpu"})
        tf = _FakeTf(["GPU:0"])

        engine._configure_tensorflow_device(tf)

        self.assertEqual(engine.active_device, "gpu")
        self.assertEqual(engine.device_name, "/GPU:0")
        self.assertEqual(tf.config.experimental.memory_growth_calls, [("GPU:0", True)])

    def test_gpu_strict_raises_when_no_gpu_visible(self) -> None:
        engine = SionnaEngine({"device": "gpu", "gpu_strict": True})

        with self.assertRaisesRegex(RuntimeError, "does not list any physical GPU"):
            engine._configure_tensorflow_device(_FakeTf([]))

    def test_auto_falls_back_to_cpu_when_no_gpu_visible(self) -> None:
        engine = SionnaEngine({"device": "auto"})

        engine._configure_tensorflow_device(_FakeTf([]))

        self.assertEqual(engine.active_device, "cpu")
        self.assertEqual(engine.device_name, "/CPU:0")

    def test_gpu_memory_limit_uses_logical_device_configuration(self) -> None:
        engine = SionnaEngine({"device": "gpu", "gpu_memory_limit_mb": 4096})
        tf = _FakeTf(["GPU:0"])

        engine._configure_tensorflow_device(tf)

        self.assertEqual(engine.active_device, "gpu")
        self.assertEqual(len(tf.config.logical_device_configurations), 1)
        _, configs = tf.config.logical_device_configurations[0]
        self.assertEqual(configs[0].memory_limit, 4096)


if __name__ == "__main__":
    unittest.main()
