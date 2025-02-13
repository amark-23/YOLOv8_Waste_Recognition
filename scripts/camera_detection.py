import cv2
from ultralytics import YOLO
import torch

def custom_object_detection():
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load your custom trained model (update path to your best.pt)
    model = YOLO(r"C:\Users\amark\yolo_dataset\runs\detect\custom_model\weights\best.pt")
    model.to(device)

    # Initialize webcam (0 for default camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)  # No need for device parameter here

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Custom YOLOv8 Detector", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    custom_object_detection()
