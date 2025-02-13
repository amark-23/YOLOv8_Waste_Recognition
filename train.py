from ultralytics import YOLO
import torch

def train_custom_yolo():
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a YOLOv8 model with pre-trained weights
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=r'C:\Users\amark\yolo_dataset\datasetconfig.yaml',  # Path to your dataset config
        epochs=100,        # Number of training epochs
        imgsz=640,         # Image size
        batch=16,          # Batch size (adjust based on GPU memory)
        patience=50,       # Early stopping patience
        device=device,     # Use GPU if available
        name='custom_model',  # Save directory for trained model
        pretrained=True,   # Use pretrained weights
        optimizer='Adam',  # Optimizer (try 'SGD', 'AdamW' for experiments)
        lr0=0.001,         # Initial learning rate
        weight_decay=0.0005,  # Regularization term
        augment=True       # Data augmentation
    )

    print("Training complete. Results saved in 'runs/detect/custom_model'.")

if __name__ == "__main__":
    train_custom_yolo()
