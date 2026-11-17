
import cv2

from app.model import ImageClassifier


image = cv2.imread("images/test.jpg")

if image is None:
    raise FileNotFoundError("Cannot read images/test.jpg")

classifier = ImageClassifier()

results = classifier.predict(image, top_k=5)

for result in results:
    print(result)
