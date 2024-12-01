import cv2
import torch
import numpy as np
import random

# Load class names from coco.names file
def load_class_names(file_path):
    with open(file_path, "r") as file:
        class_names = file.read().strip().split("\n")
    return class_names

# Load the class labels
coco_labels = load_class_names("coco.names")

# Function to generate random colors
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# Load the model from the .pt file
model_path = "ssd300_vgg16.pt"  # Your local model file
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()
print("Model loaded successfully!")

# Real-time object detection with OpenCV
def detect_and_display():
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert the frame to a tensor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            predictions = model(img_tensor)

        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter predictions with confidence above 0.5
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                # Ensure label index is within the valid range
                label_index = int(labels[i])
                if label_index < len(coco_labels):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    label = coco_labels[label_index]  # Get the class name from coco.names
                    confidence = scores[i]

                    # Generate a random color for the bounding box
                    color = generate_random_color()

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Add label with confidence
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the output frame
        cv2.imshow("Real-Time Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection function
if __name__ == "__main__":
    detect_and_display()
