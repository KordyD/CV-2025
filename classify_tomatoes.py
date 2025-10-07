import cv2
import numpy as np
import os
import glob
from dataclasses import dataclass

CATEGORY_TO_ID = {
    "fully_ripened": 1,
    "half_ripened": 2,
    "green": 3,
}

@dataclass(frozen=True)
class SegmentResult:
    mask: np.ndarray
    contour: np.ndarray | None
    red_mask: np.ndarray
    green_mask: np.ndarray
    color_mask: np.ndarray
    edge_map: np.ndarray

def segment_tomato(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must contain pixel data")

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # цветовые диапазоны
    lower_red_1 = np.array([0, 65, 40], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 65, 40], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
    lower_green = np.array([35, 15, 25], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)
    lower_sat = np.array([0, 30, 30], dtype=np.uint8)
    upper_sat = np.array([179, 255, 255], dtype=np.uint8)
    lower_value = np.array([0, 0, 40], dtype=np.uint8)
    upper_value = np.array([179, 255, 255], dtype=np.uint8)

    # создаем маски
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red_1, upper_red_1), cv2.inRange(hsv, lower_red_2, upper_red_2))
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    saturation_mask = cv2.inRange(hsv, lower_sat, upper_sat)
    value_mask = cv2.inRange(hsv, lower_value, upper_value)

    color_mask = cv2.bitwise_or(red_mask, green_mask)
    color_mask = cv2.bitwise_or(color_mask, cv2.bitwise_and(saturation_mask, value_mask))
    
    # морфологические операции, удаление шума
    color_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, color_kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, color_kernel, iterations=1)

    # детекция границ
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(gray))
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    upper = max(upper, lower + 10)
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.bitwise_and(edges, edges, mask=color_mask)

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_enhanced = cv2.dilate(edges, edge_kernel, iterations=1)
    edge_enhanced = cv2.morphologyEx(edge_enhanced, cv2.MORPH_CLOSE, edge_kernel, iterations=2)

    # сегментация
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(color_mask, fill_kernel, iterations=2)

    distance = cv2.distanceTransform(color_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.4 * distance.max(), 255.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    
    if not sure_fg.any():
        sure_fg = cv2.erode(color_mask, fill_kernel, iterations=1)
    if not sure_fg.any():
        sure_fg = color_mask.copy()

    unknown = cv2.subtract(sure_bg, sure_fg)
    num_components, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_input = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(watershed_input, markers)

    # выбираем лучшую область
    final_mask = np.zeros_like(color_mask)
    max_overlap = 0
    best_label = None
    color_bool = color_mask > 0

    max_label = int(markers.max()) if markers.size else 1
    for label in range(2, max_label + 1):
        region = markers == label
        if not np.any(region):
            continue
        overlap = np.count_nonzero(region & color_bool)
        if overlap > max_overlap:
            max_overlap = overlap
            best_label = label

    if best_label is not None and max_overlap > 0:
        final_mask[markers == best_label] = 255
    else:
        combined = cv2.bitwise_or(color_mask, edge_enhanced)
        filled = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, fill_kernel, iterations=1)
        final_mask = filled if filled.any() else color_mask.copy()

    # находим контур
    contour = None
    if final_mask.any():
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)

    if contour is not None:
        mask_from_contour = np.zeros_like(final_mask)
        cv2.drawContours(mask_from_contour, [contour], -1, 255, thickness=cv2.FILLED)
        final_mask = mask_from_contour
    else:
        final_mask = color_mask

    watershed_edges = np.zeros_like(color_mask)
    watershed_edges[markers == -1] = 255
    edge_map = cv2.bitwise_or(edge_enhanced, watershed_edges)

    return SegmentResult(
        mask=final_mask,
        contour=contour,
        red_mask=red_mask,
        green_mask=green_mask,
        color_mask=color_mask,
        edge_map=edge_map,
    )

def extract_features_from_array(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must contain pixel data")

    # полное изображение в RGB и LAB
    rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_full[..., 0] = lab_full[..., 0] / 255.0
    lab_full[..., 1:] = (lab_full[..., 1:] - 128.0) / 127.0

    # признаки полного изображения
    flat_full = rgb_full.reshape(-1, 3)
    mean_rgb_full = flat_full.mean(axis=0)
    red_dominant_full = np.mean((flat_full[:, 0] > flat_full[:, 1]) & (flat_full[:, 0] > flat_full[:, 2]))
    green_dominant_full = np.mean((flat_full[:, 1] > flat_full[:, 0]) & (flat_full[:, 1] > flat_full[:, 2]))
    blue_dominant_full = np.mean((flat_full[:, 2] > flat_full[:, 0]) & (flat_full[:, 2] > flat_full[:, 1]))
    
    channel_range_full = flat_full.max(axis=1) - flat_full.min(axis=1)
    saturation_full = channel_range_full.mean()
    value_full = flat_full.max(axis=1).mean()

    flat_lab_full = lab_full.reshape(-1, 3)
    mean_lab_full = flat_lab_full.mean(axis=0)

    # сегментация
    segment = segment_tomato(image_bgr)

    # форма объекта
    circularity = 0.0
    if segment.contour is not None:
        area = cv2.contourArea(segment.contour)
        perimeter = cv2.arcLength(segment.contour, True)
        if perimeter > 0:
            circularity = float(4.0 * np.pi * area / (perimeter**2 + 1e-6))

    # маска объекта
    object_mask = segment.mask
    mask_bool = object_mask.astype(bool)
    if not mask_bool.any():
        mask_bool = np.ones(object_mask.shape, dtype=bool)

    total_pixels = mask_bool.size
    mask_pixels = int(mask_bool.sum())
    if mask_pixels == 0:
        mask_pixels = total_pixels
    area_ratio = float(mask_pixels) / float(total_pixels)

    # HSV для объекта
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    hsv_float[..., 0] = hsv_float[..., 0] / 179.0
    hsv_float[..., 1:] = hsv_float[..., 1:] / 255.0

    # признаки внутри маски
    flat_rgb = rgb_full[mask_bool]
    flat_hsv = hsv_float[mask_bool]
    flat_lab = lab_full[mask_bool]

    if flat_rgb.size == 0:
        flat_rgb = rgb_full.reshape(-1, 3)
    if flat_hsv.size == 0:
        flat_hsv = hsv_float.reshape(-1, 3)
    if flat_lab.size == 0:
        flat_lab = lab_full.reshape(-1, 3)

    mean_rgb = flat_rgb.mean(axis=0)
    mean_hsv = flat_hsv.mean(axis=0)
    mean_lab = flat_lab.mean(axis=0)

    # стандартные отклонения
    hue_std = float(flat_hsv[:, 0].std())
    sat_std = float(flat_hsv[:, 1].std())
    lab_a_std = float(flat_lab[:, 1].std())
    lab_b_std = float(flat_lab[:, 2].std())

    # цветовые соотношения
    red_bool = segment.red_mask.astype(bool)
    green_bool = segment.green_mask.astype(bool)
    red_ratio = float(np.count_nonzero(red_bool & mask_bool)) / float(mask_pixels)
    green_ratio = float(np.count_nonzero(green_bool & mask_bool)) / float(mask_pixels)

    return np.array([
        mean_rgb_full[0], mean_rgb_full[1], mean_rgb_full[2],
        red_dominant_full, green_dominant_full, blue_dominant_full,
        saturation_full, value_full,
        mean_lab_full[0], mean_lab_full[1], mean_lab_full[2],
        mean_rgb[0], mean_rgb[1], mean_rgb[2],
        mean_hsv[0], mean_hsv[1], mean_hsv[2],
        mean_lab[0], mean_lab[1], mean_lab[2],
        red_ratio, green_ratio, area_ratio, circularity,
        hue_std, sat_std, lab_a_std, lab_b_std,
    ], dtype=np.float32)

def extract_features(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return extract_features_from_array(image_bgr)

def compute_centroids(features, labels):
    centroids = {}
    for label_id in np.unique(labels):
        class_features = features[labels == label_id]
        centroids[label_id] = class_features.mean(axis=0)
    return centroids

def predict_label(feature, centroids):
    mean_hue = float(feature[14])
    red_ratio = float(feature[20])
    
    if mean_hue >= 0.16 and red_ratio < 0.15:
        return CATEGORY_TO_ID["green"]
    
    distances = {}
    for label_id, centroid in centroids.items():
        distances[label_id] = np.linalg.norm(feature - centroid)
    
    return min(distances.keys(), key=lambda k: distances[k])

def load_training_data():
    records = []
    for category, label in CATEGORY_TO_ID.items():
        category_dir = f"train/{category}"
        if not os.path.exists(category_dir):
            print(f"Warning: {category_dir} directory not found")
            continue
        
        image_files = glob.glob(f"{category_dir}/*.jpg")
        for image_path in sorted(image_files):
            records.append({
                "path": image_path,
                "category": category,
                "label_id": label
            })
    
    features = []
    labels = []
    
    for record in records:
        try:
            feature_vector = extract_features(record["path"])
            features.append(feature_vector)
            labels.append(record["label_id"])
        except:
            pass
    
    return np.vstack(features), np.array(labels, dtype=np.int32)



def classify_test_images(centroids):
    print("Starting classification...")
    test_images = glob.glob("test/*.jpg")
    test_images.sort()
    
    if not test_images:
        print("Error: No test images found")
        return
    
    results = []
    for img_path in test_images:
        filename = os.path.basename(img_path)
        try:
            features = extract_features(img_path)
            predicted = predict_label(features, centroids)
            results.append((filename, predicted))
            print(f"{filename} -> category {predicted}")
        except:
            results.append((filename, 3))
            print(f"{filename} -> category 3 (default)")
    
    with open("answer.txt", "w") as f:
        for filename, category in results:
            f.write(f"{filename} - {category}\n")
    
    print("Classification completed. Results saved to answer.txt")

def main():
    if not os.path.exists("train"):
        print("Error: train directory not found")
        return
    if not os.path.exists("test"):
        print("Error: test directory not found")
        return
    
    features, labels = load_training_data()
    centroids = compute_centroids(features, labels)
    classify_test_images(centroids)

if __name__ == "__main__":
    main()