from ultralytics import YOLO
import torch

def evaluate_custom_yolo():
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load your trained model (replace with actual path to best.pt)
    model = YOLO(r"C:\Users\amark\yolo_dataset\runs\detect\custom_model\weights\best.pt")

    # Evaluate the model on the test set
    results = model.val(
        data=r"C:\Users\amark\yolo_dataset\datasetconfig.yaml",  # Path to YAML file
        device=device,
        split='test',  # Use the test set
        imgsz=640,
        batch=16,
        conf=0.25,  # Confidence threshold
        iou=0.6  # NMS IOU threshold
    )

    # Print evaluation results
    print("Evaluation Results:")
    print(results)

    # Access specific metrics
    mAP50 = results.box.map50  # Mean Average Precision at IoU 0.5
    mAP50_95 = results.box.map  # Mean Average Precision at IoU 0.5:0.95
    precision = results.box.mp  # Model Precision
    recall = results.box.mr  # Model Recall

    print(f"mAP50: {mAP50:.4f}")
    print(f"mAP50-95: {mAP50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    evaluate_custom_yolo()
