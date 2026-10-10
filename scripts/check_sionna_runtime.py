"""Print TensorFlow/Sionna runtime status for the O-RAN simulator."""

from __future__ import annotations

import os


def main() -> None:
    print(f"DRJIT_LIBLLVM_PATH={os.environ.get('DRJIT_LIBLLVM_PATH', '')}")
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        print(f"tensorflow_import_error={type(exc).__name__}: {exc}")
        return

    print(f"tensorflow_version={tf.__version__}")
    print(f"tensorflow_gpus={tf.config.list_physical_devices('GPU')}")

    try:
        import sionna
    except Exception as exc:  # pragma: no cover
        print(f"sionna_import_error={type(exc).__name__}: {exc}")
        return

    print(f"sionna_version={getattr(sionna, '__version__', 'unknown')}")


if __name__ == "__main__":
    main()
