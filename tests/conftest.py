import pytest
import numpy as np
import cv2

@pytest.fixture
def sample_image():
    image = np.full((100, 200, 3), 128, dtype=np.uint8)

    success, encoded = cv2.imencode(".png", image)
    assert success

    return image, encoded.tobytes()