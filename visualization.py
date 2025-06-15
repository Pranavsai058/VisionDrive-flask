import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
MODEL_PATH = 'ml_model/multi_stage_edge_detection_model.keras'
IMG_SIZE = (224, 224)

# Custom metric for model loading
def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)

# Load model (only once)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'f1_metric': f1_metric}
)

def load_and_preprocess(img_path):
    """Load and preprocess an image for model input"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype('float32') / 255.0

def analyze_navigation(edge_map):
    """Analyze edge map to determine navigation direction"""
    h, w = edge_map.shape
    regions = {
        'left': edge_map[:, :w//3],
        'center': edge_map[h//2:, w//3:2*w//3],
        'right': edge_map[:, 2*w//3:]
    }
    densities = {k: np.mean(v) for k, v in regions.items()}

    # Thresholds (tuned for better decision making)
    margin = 0.07
    clear_threshold = 0.15
    block_threshold = 0.38

    center_density = densities['center']
    left_density = densities['left']
    right_density = densities['right']

    if center_density < clear_threshold and center_density < left_density and center_density < right_density:
        return "clear path", densities
    if center_density > block_threshold:
        return "blocked", densities
    if left_density + margin < right_density:
        return "left", densities
    elif right_density + margin < left_density:
        return "right", densities
    else:
        return "blocked", densities

def create_navigation_display(original_img, edge_map, direction, densities):
    """Create visualization with navigation overlay"""
    # Convert edge map to RGB and resize to original dimensions
    edge_rgb = np.stack([edge_map]*3, axis=2)
    edge_rgb = (edge_rgb * 255).astype(np.uint8)
    original_h, original_w = original_img.shape[:2]
    edge_rgb = cv2.resize(edge_rgb, (original_w, original_h))
    
    # Combine with original image
    combined = cv2.addWeighted(original_img, 0.7, edge_rgb, 0.3, 0)
    
    # Convert to PIL for drawing
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    h, w = original_h, original_w

    # Draw navigation indicators based on direction
    if direction in ["left", "right"]:
        arrow_length = 60
        line_width = 6
        arrow_color = (0, 255, 0)  # Green
        start_point = (w//2, h-50)
        if direction == "left":
            end_point = (w//2 - arrow_length, h-100)
        else:
            end_point = (w//2 + arrow_length, h-100)

        draw.line([start_point, end_point], fill=arrow_color, width=line_width)
        head_size = 15
        if direction == "left":
            head_points = [
                (end_point[0] + head_size, end_point[1] - head_size),
                end_point,
                (end_point[0] + head_size, end_point[1] + head_size)
            ]
        else:
            head_points = [
                (end_point[0] - head_size, end_point[1] - head_size),
                end_point,
                (end_point[0] - head_size, end_point[1] + head_size)
            ]
        draw.polygon(head_points, fill=arrow_color)

    elif direction == "clear path":
        arrow_length = 60
        arrow_color = (0, 255, 0)  # Green
        start_point = (w//2, h-50)
        end_point = (w//2, h-100)
        draw.line([start_point, end_point], fill=arrow_color, width=6)
        draw.polygon([
            (end_point[0] - 10, end_point[1] + 15),
            (end_point[0] + 10, end_point[1] + 15),
            (end_point[0], end_point[1] - 10)
        ], fill=arrow_color)

    else:  # blocked
        size = 40
        center_x, center_y = w//2, h//2
        draw.line([(center_x - size, center_y - size), (center_x + size, center_y + size)], 
                 fill=(255,0,0), width=5)  # Red X
        draw.line([(center_x + size, center_y - size), (center_x - size, center_y + size)], 
                 fill=(255,0,0), width=5)
        
        # Add warning text using OpenCV
        combined = np.array(img)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.putText(combined, "ROAD BLOCKED!", (w//2-120, h//2-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(combined, "TAKE U-TURN", (w//2-110, h//2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return combined, direction

    # Convert back to OpenCV format for saving
    result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return result, direction

def process_single_image(rgb_path, lidar_path, output_path):
    """Process a single pair of RGB and LiDAR images"""
    # Load and prepare RGB image
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        raise ValueError("Could not read RGB image")
    
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb_img, IMG_SIZE)
    processed_rgb = (resized_rgb.astype('float32') / 255.0)[np.newaxis, ...]
    
    # Load and prepare LiDAR image
    if os.path.exists(lidar_path):
        lidar_img = cv2.imread(lidar_path)
        if lidar_img is not None:
            lidar_img = cv2.cvtColor(lidar_img, cv2.COLOR_BGR2RGB)
            resized_lidar = cv2.resize(lidar_img, IMG_SIZE)
            processed_lidar = (resized_lidar.astype('float32') / 255.0)[np.newaxis, ...]
        else:
            processed_lidar = np.zeros((1, *IMG_SIZE, 3), dtype='float32')
    else:
        processed_lidar = np.zeros((1, *IMG_SIZE, 3), dtype='float32')
    
    # Predict edges
    pred = model.predict([processed_rgb, processed_lidar], verbose=0)[0]
    edges = (pred[..., 0] > 0.3).astype(np.uint8) * 255
    
    # Analyze navigation
    direction, densities = analyze_navigation(pred[..., 0])
    
    # Create and save visualization
    result, direction = create_navigation_display(rgb_img, pred[..., 0], direction, densities)
    cv2.imwrite(output_path, result)
    
    return direction