from fastapi.testclient import TestClient
from app.main import app
import cv2
import numpy as np
import pytest

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Industrial AI Platform is running!"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image():
    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode('.png', image)
    assert success

    response = client.post("/analyze", files={"file": ("test_image.png", encoded.tobytes(), "image/png")})
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