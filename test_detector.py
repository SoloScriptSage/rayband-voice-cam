import dlib
import cv2

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Load a test image (replace 'test.jpg' with your image path)
image_path = "test.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Could not load image at {image_path}")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    print(f"Number of faces detected: {len(faces)}")
    for i, face in enumerate(faces):
        print(f"Face {i+1}: Left:{face.left()} Top:{face.top()} Right:{face.right()} Bottom:{face.bottom()}")
