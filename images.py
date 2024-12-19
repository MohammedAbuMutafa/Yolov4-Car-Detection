from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import subprocess
import glob
import time
import matplotlib.pyplot as plt
import re

def parse_yolo_label(label_path):
    """
    Parse YOLO label format from a .txt file and return bounding boxes and classes.
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        values = list(map(float, line.split()))
        class_id = int(values[0])
        x_center, y_center, width, height = values[1:]
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def calculate_metrics(ground_truth, predictions, num_classes=1, iou_threshold=0.5):
    """
    Calculate precision, recall, F1 score, and accuracy based on YOLO outputs.
    """
    y_true = []
    y_pred = []
    
    for gt_boxes, pred_boxes in zip(ground_truth, predictions):
        gt_classes = [box[0] for box in gt_boxes]
        pred_classes = [box[0] for box in pred_boxes]
        
        # Append class presence for each image
        for c in range(num_classes):
            y_true.append(1 if c in gt_classes else 0)
            y_pred.append(1 if c in pred_classes else 0)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1

# --------------------------
# Helper Function: Absolute Paths
# --------------------------

def get_absolute_path(relative_path):
    """
    Convert a relative path to an absolute path.
    """
    return os.path.abspath(relative_path)

# --------------------------
# Step 1: Import the Dataset
# --------------------------

def import_dataset(dataset_path):
    """
    Imports images from a dataset folder.
    """
    dataset_path = get_absolute_path(dataset_path)
    image_files = [
        os.path.join(dataset_path, img)
        for img in os.listdir(dataset_path)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Step 1: Imported {len(image_files)} images from {dataset_path}")
    return image_files

# --------------------------
# Step 2: Split Dataset
# --------------------------

def split_dataset(image_files, test_size=0.2, val_size=0.2, output_base_dir="/Users/yousef/Desktop/vehicleImages"):
    """
    Splits the dataset into train, validation, and test sets, and copies corresponding labels.
    """
    output_base_dir = get_absolute_path(output_base_dir)
    annotations_dir = "/Users/yousef/Desktop/vehicleImages/annotation/labels"
    
    # Debug: Print the paths we're using
    print(f"Looking for labels in: {annotations_dir}")
    
    # Create directories if they don't exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)

    # Split the dataset
    train_files, temp_files = train_test_split(image_files, test_size=test_size + val_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_size / (test_size + val_size), random_state=42)

    def copy_files_and_labels(file_list, split_name):
        images_dir = os.path.join(output_base_dir, split_name, 'images')
        labels_dir = os.path.join(output_base_dir, split_name, 'labels')
        
        for image_path in file_list:
            # Copy image
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(images_dir, image_name))
            
            # Get corresponding label file path
            label_name = os.path.splitext(image_name)[0] + '.txt'
            
            # Look for label in both train and test folders
            possible_label_paths = [
                os.path.join(annotations_dir, 'train', label_name),
                os.path.join(annotations_dir, 'test', label_name)
            ]
            
            label_found = False
            for label_src in possible_label_paths:
                if os.path.exists(label_src):
                    shutil.copy(label_src, os.path.join(labels_dir, label_name))
                    label_found = True
                    break
            
            if not label_found:
                print(f"Warning: No label file found for {image_name}")
                # Create an empty label file to prevent YOLO from crashing
                with open(os.path.join(labels_dir, label_name), 'w') as f:
                    pass

    # Copy files and labels for each split
    print("\nProcessing train split...")
    copy_files_and_labels(train_files, 'train')
    print("\nProcessing validation split...")
    copy_files_and_labels(val_files, 'val')
    print("\nProcessing test split...")
    copy_files_and_labels(test_files, 'test')

    # Verify label files exist
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(output_base_dir, split, 'images')
        labels_dir = os.path.join(output_base_dir, split, 'labels')
        image_count = len(os.listdir(images_dir))
        label_count = len(os.listdir(labels_dir))
        print(f"\n{split} set:")
        print(f"Images: {image_count}")
        print(f"Labels: {label_count}")

    return (os.path.join(output_base_dir, 'train'),
            os.path.join(output_base_dir, 'val'),
            os.path.join(output_base_dir, 'test'))

# --------------------------
# Step 3: Train, Validate, and Test Functionality
# --------------------------

def parse_yolo_output(line):
    """
    Parse YOLO training output to extract loss values.
    """
    loss_match = re.search(r'avg loss: (\d+\.?\d*)', line)
    if loss_match:
        return float(loss_match.group(1))
    return None

def train_yolo(config_path, data_file_path, weights_output_path):
    """
    Modified train_yolo function with improved error handling and output capture
    """
    darknet_path = "/Users/yousef/Desktop/vehicleImages/darknet-master/build/darknet"
    initial_weights = "/Users/yousef/Desktop/vehicleImages/darknet-master/yolov4.conv.137"
    
    # Initialize loss tracking
    training_losses = []
    current_batch_losses = []
    iterations = []
    current_iteration = 0
    
    # Initialize lists to track metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    # Verify paths and permissions
    for path, name in [
        (darknet_path, "Darknet executable"),
        (config_path, "Config file"),
        (data_file_path, "Data file"),
        (initial_weights, "Initial weights")
    ]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            return False
        if name == "Darknet executable" and not os.access(path, os.X_OK):
            print(f"Error: {name} is not executable. Running chmod +x...")
            try:
                os.chmod(path, 0o755)
            except Exception as e:
                print(f"Failed to make darknet executable: {e}")
                return False

    # Ensure backup directory exists
    os.makedirs(weights_output_path, exist_ok=True)

    # Command to run
    command = [
        darknet_path,
        "detector",
        "train",
        data_file_path,
        config_path,
        initial_weights,
        "-dont_show",
        "-map"
    ] 

    print("\nStarting YOLO training with command:")
    print(" ".join(command))
    print("\nVerifying file contents:")
    print(f"\nData file ({data_file_path}):")
    try:
        with open(data_file_path, 'r') as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading data file: {e}")
    
    try:
        # Redirect output to a log file
        with open("/Users/yousef/Desktop/vehicleImages/darknet-master/training_log.txt", "w") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,   # Capture standard error
                universal_newlines=True,
                cwd=os.path.dirname(darknet_path)
            )
            
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Check for loss and metrics in the output
                loss = parse_yolo_output(line)
                if loss is not None:
                    current_batch_losses.append(loss)
                    # Calculate metrics (you may need to adjust this part based on your output)
                    accuracy, precision, recall, f1 = calculate_metrics(ground_truth, predictions)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    iterations.append(len(iterations) + 1)  # Increment iteration count

            # Check if process completed successfully
            _, stderr = process.communicate()
            if process.returncode != 0:
                print(f"\nTraining failed with return code {process.returncode}")
                if stderr:
                    print("\nError output:")
                    print(stderr)
                return False
            
            print("\nTraining completed successfully!")
            # Plot metrics after training
            plot_metrics(iterations, precisions, recalls, f1_scores)
            return True
            
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def plot_training_loss(iterations, losses, save_path=None):
    """
    Plot and optionally save the training loss graph
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'b-', label='Training Loss')
    plt.title('YOLO Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.draw()
        plt.pause(0.1)

def validate_yolo(config_path, data_file_path, weights_path, val_dir):
    """
    Modified validate_yolo function to track mAP
    """
    validation_results = []
    
    command = [
        "/Users/yousef/Desktop/vehicleImages/darknet-master/build/darknet",
        "detector", "map",
        data_file_path,
        config_path,
        weights_path
    ]
    
    print(f"Validating YOLO model on {val_dir}...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Parse mAP from output
    output = stdout.decode("utf-8")
    map_match = re.search(r'mean average precision \(mAP@0\.50\) = (\d+\.?\d*)', output)
    if map_match:
        map_value = float(map_match.group(1))
        validation_results.append(map_value)
        
        # Plot validation results
        plt.figure(figsize=(10, 6))
        plt.plot(validation_results, 'r-', label='Validation mAP')
        plt.title('YOLO Validation mAP')
        plt.xlabel('Validation Check')
        plt.ylabel('mAP@0.50')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(weights_path), 'validation_map.png'))
        plt.close()
    
    print(output)
    print(stderr.decode("utf-8"))

def test_yolo_with_metrics(config_path, data_file_path, weights_path, test_dir, output_dir="test_results"):
    """
    Tests the YOLO model, saves predictions, and calculates evaluation metrics.
    """
    darknet_exe = "/Users/yousef/Desktop/vehicleImages/darknet-master/build/darknet"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        os.path.join(test_dir, img)
        for img in os.listdir(test_dir)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    ground_truth = []
    predictions = []

    for img_path in image_files:
        # Generate prediction file
        command = [
            darknet_exe,
            "detector", "test",
            data_file_path,
            config_path,
            weights_path,
            img_path,
            "-dont_show"
        ]
        subprocess.run(command)

        # Parse predictions
        prediction_file = "/Users/yousef/Desktop/vehicleImages/darknet-master/predictions.txt"
        prediction_boxes = parse_yolo_label(prediction_file)
        predictions.append(prediction_boxes)

        # Parse ground truth
        gt_label_file = os.path.join(test_dir.replace('images', 'labels'), os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        gt_boxes = parse_yolo_label(gt_label_file)
        ground_truth.append(gt_boxes)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(ground_truth, predictions)

    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save metrics to a file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

def generate_labels(pretrained_weights, config_path, data_file_path, image_dir, labels_dir):
    """
    Generates YOLO-format labels for a set of images using a pretrained YOLO model.
    """
    darknet_exe = "/Users/yousef/Desktop/vehicleImages/darknet-master/build/darknet"
    obj_names_path = "/Users/yousef/Desktop/vehicleImages/data/obj.names"  # Updated absolute path
    
    # Ensure the obj.names file exists
    os.makedirs(os.path.dirname(obj_names_path), exist_ok=True)
    if not os.path.exists(obj_names_path):
        with open(obj_names_path, 'w') as f:
            f.write("vehicle\n")
    
    command = [
        darknet_exe,
        "detector",
        "test",
        data_file_path,
        config_path,
        pretrained_weights,
        "-dont_show"
    ]
    # Rest of the function remains the same

def create_dataset_files():
    dataset_path = "/Users/yousef/Desktop/vehicleImages/"
    # Change this to look in both train and test subdirectories of annotation/labels
    annotation_labels_path = os.path.join(dataset_path, "annotation", "labels")
    
    # Function to copy labels for split
    def copy_labels_for_split(split_name):
        images_dir = os.path.join(dataset_path, split_name, "images")
        labels_dir = os.path.join(dataset_path, split_name, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Use glob to find all image types
        image_patterns = ['.jpg', '.jpeg', '*.png']
        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob.glob(os.path.join(images_dir, pattern)))
        
        valid_images = []
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            
            # Look for labels in both train and test subdirectories
            possible_label_paths = [
                os.path.join(annotation_labels_path, "train", label_name),
                os.path.join(annotation_labels_path, "test", label_name)
            ]
            
            label_found = False
            for src_label_path in possible_label_paths:
                if os.path.exists(src_label_path):
                    dst_label_path = os.path.join(labels_dir, label_name)
                    shutil.copy2(src_label_path, dst_label_path)
                    valid_images.append(img_path)
                    label_found = True
                    break
            
            if not label_found:
                print(f"Warning: Missing label for {img_name}")
        
        return valid_images

    # Process splits and create txt files
    print("\nProcessing train split...")
    train_images = copy_labels_for_split("train")
    
    print("\nProcessing validation split...")
    val_images = copy_labels_for_split("val")
    
    # Create train.txt and val.txt with absolute paths
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        for img_path in train_images:
            f.write(f"{os.path.abspath(img_path)}\n")
    
    with open(os.path.join(dataset_path, "val.txt"), "w") as f:
        for img_path in val_images:
            f.write(f"{os.path.abspath(img_path)}\n")

    print("\nDataset statistics:")
    print(f"Train images with labels: {len(train_images)}")
    print(f"Validation images with labels: {len(val_images)}")

def create_config_files():
    # Use raw string to handle spaces correctly
    base_path = r"/Users/yousef/Desktop/vehicleImages"
    data_dir = os.path.join(base_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create obj.names with explicit path
    obj_names_path = os.path.join(data_dir, "obj.names")
    with open(obj_names_path, "w") as f:
        f.write("vehicle\n")
    print(f"Created obj.names at: {obj_names_path}")
    
    # Create obj.data with escaped paths
    obj_data_path = os.path.join(data_dir, "obj.data")
    with open(obj_data_path, "w") as f:
        f.write(f"classes = 1\n")
        f.write(f"train = {os.path.join(base_path, 'train.txt')}\n")
        f.write(f"valid = {os.path.join(base_path, 'val.txt')}\n")
        f.write(f"names = {obj_names_path}\n")
        f.write(f"backup = {os.path.join(base_path, 'backup')}\n")
    print(f"Created obj.data at: {obj_data_path}")

    # Create backup directory
    backup_dir = os.path.join(base_path, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory at: {backup_dir}")

    # Print the actual content of obj.names to verify
    print("\nContents of obj.names:")
    with open(obj_names_path, 'r') as f:
        print(f.read())

    # Print the actual content of obj.data to verify
    print("\nContents of obj.data:")
    with open(obj_data_path, 'r') as f:
        print(f.read())

def create_data_file():
    """
    Creates and returns the path to the data file
    """
    base_dir = "/Users/yousef/Desktop/vehicleImages"
    data_path = os.path.join(base_dir, "data", "obj.data")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Write the data file
    with open(data_path, "w") as f:
        f.write(f"classes = 1\n")
        f.write(f"train = {os.path.join(base_dir, 'train.txt')}\n")
        f.write(f"valid = {os.path.join(base_dir, 'val.txt')}\n")
        f.write(f"names = {os.path.join(base_dir, 'data', 'obj.names')}\n")
        f.write(f"backup = {os.path.join(base_dir, 'backup')}\n")
    
    return data_path

def create_label_file(image_path, detections, labels_dir):
    """
    Create a YOLO format label file for an image
    """
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create label file path (same name as image but .txt extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(labels_dir, f"{base_name}.txt")
    
    with open(label_path, 'w') as f:
        for detection in detections:
            # Convert bbox to YOLO format
            class_id = 0  # 0 for vehicle
            x_center = (detection['x'] + detection['width']/2) / width
            y_center = (detection['y'] + detection['height']/2) / height
            w = detection['width'] / width
            h = detection['height'] / height
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def create_prediction_video(config_path, weights_path, data_file_path, images_dir, output_video_path, fps=10):
    """
    Creates a video from images using Darknet's YOLO detection.
    Uses the exact command structure that works manually.
    """
    # Get all images from the directory
    image_files = sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if not image_files:
        print("No images found in the specified directory")
        return

    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i + 1}/{len(image_files)}: {os.path.basename(img_path)}")  # Debugging output
        # Use the exact command that works manually
        command = f"cd /Users/yousef/Desktop/vehicleImages/darknet-master && ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights {img_path}"
        
        os.system(command)  # Use os.system instead of subprocess
        
        # Read the predictions image
        predictions_path = "/Users/yousef/Desktop/vehicleImages/darknet-master/predictions.jpg"
        if os.path.exists(predictions_path):
            frame = cv2.imread(predictions_path)
            if frame is not None:
                # Write frame to video
                out.write(frame)
            # Clean up predictions file
            os.remove(predictions_path)
        else:
            print(f"Warning: No predictions generated for {os.path.basename(img_path)}")
            # Use original frame if no predictions
            frame = cv2.imread(img_path)
            out.write(frame)

        # Give time for the window to display
        cv2.waitKey(100)

    # Release resources
    out.release()
    print(f"\nVideo saved to {output_video_path}")

def run_yolo_detection(delay_seconds=2):
    """
    Runs YOLO detection on images in train, test, and validation sets using os.system
    Shows detection window for each image with a delay between images
    """
    # Directories to process
    dirs_to_process = ['train', 'val', 'test']
    
    for dir_name in dirs_to_process:
        print(f"\nProcessing {dir_name} directory...")
        images_dir = f"/Users/yousef/Desktop/vehicleImages/{dir_name}/images"
        
        # Get all images in the directory
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(images_dir, image_file)
            print(f"\nProcessing image {i}/{len(image_files)}: {image_file}")
            
            # Use the exact command structure that works for you
            command = f"cd /Users/yousef/Desktop/vehicleImages/darknet-master && "
            command += f"./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights "
            command += f"{image_path}"
            
            # Execute the command
            print(f"Executing command: {command}")
            os.system(command)
            
            # Add delay between images
            time.sleep(delay_seconds)

def monitor_training(log_file_path="/Users/yousef/Desktop/vehicleImages/darknet-master/training_log.txt"):
    """
    Actively monitors training progress and creates plots in real-time
    """
    print("Starting training monitoring...")
    
    # Create output directory
    output_dir = "/Users/yousef/Desktop/vehicleImages/training_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    losses = []
    iterations = []
    
    try:
        while True:  # Keep monitoring until interrupted
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()
                
                # Clear previous data
                losses = []
                iterations = []
                
                # Parse log file
                for line in lines:
                    if 'avg loss' in line:
                        loss_match = re.search(r'avg loss: (\d+\.?\d*)', line)
                        if loss_match:
                            loss = float(loss_match.group(1))
                            losses.append(loss)
                            iterations.append(len(losses))
                
                if losses:  # Only plot if we have data
                    plt.figure(figsize=(10, 5))
                    plt.plot(iterations, losses, 'b-', label='Training Loss')
                    plt.title('YOLO Training Loss Over Time')
                    plt.xlabel('Iterations')
                    plt.ylabel('Average Loss')
                    plt.grid(True)
                    plt.legend()
                    
                    # Save plot
                    plot_path = os.path.join(output_dir, 'training_loss_current.png')
                    plt.savefig(plot_path)
                    plt.close()
                    
                    print(f"\rCurrent loss: {losses[-1]:.4f} (Iteration {len(losses)})", end='')
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nStopped monitoring training progress")
        if losses:
            print(f"Final loss: {losses[-1]:.4f}")

def plot_metrics(iterations, precisions, recalls, f1_scores):
    """
    Plot precision, recall, and F1 score over iterations
    """
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, precisions, label='Precision', color='blue')
    plt.plot(iterations, recalls, label='Recall', color='green')
    plt.plot(iterations, f1_scores, label='F1 Score', color='red')
    plt.title('Precision, Recall, and F1 Score Over Training Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.savefig('/Users/yousef/Desktop/vehicleImages/training_metrics.png')
    plt.close()

# --------------------------
# Main Workflow
# --------------------------

if __name__ == "__main__":
    base_dir = "/Users/yousef/Desktop/vehicleImages"
    
    # Verify base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        exit(1)
    
    print("Creating dataset files...")
    create_dataset_files()
    
    print("Creating configuration files...")
    create_config_files()
    
    # Setup paths
    config_path = f"{base_dir}/darknet-master/cfg/yolov4.cfg"
    weights_path = f"{base_dir}/darknet-master/yolov4.conv.137"
    data_file_path = create_data_file()
    
    # Verify all required files exist
    required_files = {
        "Config file": config_path,
        "Weights file": weights_path,
        "Data file": data_file_path
    }
    
    missing_files = False
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            missing_files = True
    
    if missing_files:
        print("Please ensure all required files exist before training")
        exit(1)
    
    # Additional verification before training
    print("\nVerifying critical files and paths:")
    critical_paths = {
        "Darknet executable": "/Users/yousef/Desktop/vehicleImages/darknet-master/build/darknet",
        "Config file": config_path,
        "Data file": data_file_path,
        "Initial weights": weights_path,
        "Training images dir": os.path.join(base_dir, "train", "images"),
        "Training labels dir": os.path.join(base_dir, "train", "labels")
    }
    
    all_paths_valid = True
    for name, path in critical_paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} ({path})")
        if not exists:
            all_paths_valid = False
    
    if not all_paths_valid:
        print("\nError: Some required files or directories are missing. Please verify paths.")
        exit(1)
    
    # Continue with training if all paths are valid
    print("\nStarting YOLO training...")
    training_success = train_yolo(config_path, data_file_path, f"{base_dir}/backup")
    
    if training_success:
        print("\nRunning final validation...")
        validate_yolo(
            config_path=config_path,
            data_file_path=data_file_path,
            weights_path=f"{base_dir}/backup/yolov4_final.weights",
            val_dir=val_dir
        )
    
    # After training completes
    if training_success:
        print("\nGenerating training progress plots...")
        log_file_path = "/Users/yousef/Desktop/vehicleImages/darknet-master/training_log.txt"
        output_dir = "/Users/yousef/Desktop/vehicleImages/training_plots"
        os.makedirs(output_dir, exist_ok=True)
        monitor_training(log_file_path, output_dir)
    
    # Add this section at the end
    print("\nDo you want to create a prediction video? (yes/no)")
    create_video = input().lower().strip()
    
    if create_video == 'yes':
        base_dir = "/Users/yousef/Desktop/vehicleImages"
        config_path = f"{base_dir}/darknet-master/cfg/yolov4.cfg"
        weights_path = f"{base_dir}/backup/yolov4_final.weights"  # Use your final trained weights
        data_file_path = f"{base_dir}/data/obj.data"
        test_images_dir = f"{base_dir}/test/images"
        output_video = f"{base_dir}/vehicle_predictions.mp4"
        
        create_prediction_video(
            config_path=config_path,
            weights_path=weights_path,
            data_file_path=data_file_path,
            images_dir=test_images_dir,
            output_video_path=output_video,
            fps=10
        )

    if training_success:
        print("\nRunning final validation and evaluation...")
        test_dir = os.path.join(base_dir, "test", "images")
        test_yolo_with_metrics(
            config_path=config_path,
            data_file_path=data_file_path,
            weights_path=f"{base_dir}/backup/yolov4_final.weights",
            test_dir=test_dir,
            output_dir=os.path.join(base_dir, "test_results")
        )
    
    print("\nDo you want to run YOLO detection on all images? (yes/no)")
    run_detection = input().lower().strip()
    
    if run_detection == 'yes':
        print("\nEnter delay between images (in seconds, default is 2):")
        try:
            delay = float(input().strip() or "2")
        except ValueError:
            delay = 2
        run_yolo_detection(delay)
    