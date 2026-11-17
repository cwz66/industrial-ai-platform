import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client =TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"

@pytest.mark.integration
def test_real_resnet_prediction():

    assert IMAGE_PATH.exists(), "Test image not found"

    image_bytes = IMAGE_PATH.read_bytes()

    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200

    results = response.json()

    assert results["filename"] == "test.jpg"
    assert len(results["predictions"]) == 5

    for prediction in results["predictions"]:
        assert isinstance(prediction["class"], str)
        assert isinstance(prediction["confidence"], float)
        assert 0.0 <= prediction["confidence"] <= 1.0


