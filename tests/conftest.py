from pathlib import Path

import pytest

from gamst_capture.synthetic import generate_synthetic


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fixture") / "coin.mp4"
    generate_synthetic(path)
    return path
