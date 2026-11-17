from fastapi.testclient import TestClient
from app.main import app
import cv2
import numpy as np
import pytest
from unittest.mock import patch

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Industrial AI Platform is running!"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image(sample_image):
    _, encoded = sample_image

    response = client.post("/analyze", files={"file": ("test_image.png", encoded, "image/png")})
    assert response.status_code == 200 

    result = response.json()

    assert result["width"] == 200
    assert result["height"] == 100
    assert result["mean_brightness"] == 128.0

def test_invalid_image():
    response = client.post("/analyze", files={"file": ("invalid.png", b"this is not an image", "image/png")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file."

def test_unsupported_file():
    response = client.post("/analyze", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file type. Only JPEG and PNG are allowed."

def test_large_image():
    response = client.post("/analyze", files={"file": ("large_image.png", b"a" * (5 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 413
    assert response.json()["detail"] == "File size exceeds the maximum limit of 5MB."

@pytest.mark.parametrize("file_size", [5 * 1024 * 1024 - 1, 5 * 1024 * 1024])
def test_image_within_size_limit(file_size):
    response = client.post("/analyze", files={"file": ("small_image.png", b"a" * file_size, "image/png")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file."

def test_predict_image(sample_image):
    _, image_bytes = sample_image

    with patch('app.main.get_classifier') as mock_get_classifier:
        fake_predictions = [{"class": "tabby", "confidence": 0.8}, {"class": "tiger cat", "confidence": 0.2}]
        mock_get_classifier.return_value.predict.return_value = fake_predictions

        response = client.post("/predict", files={"file": ("test.png", image_bytes, "image/png")})

        assert response.status_code == 200

        result = response.json()

        assert result["filename"] == "test.png"
        assert result["predictions"] == fake_predictions

        mock_get_classifier.assert_called_once()

def test_predict_image_arguments(sample_image):
    image, image_bytes = sample_image

    with patch('app.main.get_classifier') as mock_get_classifier:
        fake_classifier = mock_get_classifier.return_value
        fake_classifier.predict.return_value = [{"class": "tabby", "confidence": 0.8}]

        response = client.post("/predict", files={"file": ("test.png", image_bytes, "image/png")})

        assert response.status_code == 200

        mock_get_classifier.assert_called_once()
        fake_classifier.predict.assert_called_once()

        args = fake_classifier.predict.call_args.args
        kwargs = fake_classifier.predict.call_args.kwargs

        received_image = args[0]

        assert isinstance(received_image, np.ndarray)
        assert received_image.shape == (100, 200, 3)
        assert received_image.dtype == np.uint8
        np.testing.assert_array_equal(received_image, image)

        assert kwargs["top_k"] == 5