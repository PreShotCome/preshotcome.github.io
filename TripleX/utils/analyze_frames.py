import json
import logging
import os

import cv2
import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from extract_sharpest_frame import extract_sharpest_frame


def load_frozen_graph(model_filename):
    logger.info(f'Loading model from {model_filename}...')
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
    logger.info(f'Model {model_filename} loaded successfully.')
    return graph


def print_tensor_names(graph):
    tensor_names = [tensor.name for op in graph.get_operations() for tensor in op.values()]
    for name in tensor_names:
        print(name)


def preprocess_image(image_path, input_size):
    try:
        # Load the image
        image = cv2.imread(image_path)
        # Resize the image to the expected input size
        image_resized = cv2.resize(image, input_size)
        # Convert to float32 (or the data type expected by your model)
        image_np = image_resized.astype(np.float32)
        # If your model expects normalized pixel values, uncomment the next line
        # image_np /= 255.0
        # Expand dimensions to match model's input shape: [batch size, height, width, channels]
        image_np = np.expand_dims(image_np, axis=0)
        return image_np
    except Exception as e:
        logger.error(f'Error preprocessing image {image_path}: {e}', exc_info=True)
        raise


def run_classification_inference(graph, image_np):
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('data:0')
            output_tensor = graph.get_tensor_by_name('model_output:0')
            predictions = sess.run(output_tensor, feed_dict={input_tensor: image_np})
    return predictions


def run_object_detection_inference(graph, image_np):
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes_tensor = graph.get_tensor_by_name('detected_boxes:0')
            scores_tensor = graph.get_tensor_by_name('detected_scores:0')
            classes_tensor = graph.get_tensor_by_name('detected_classes:0')
            feed_dict = {input_tensor: image_np}
            boxes, scores, classes = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor],
                feed_dict=feed_dict
            )
    return boxes, scores, classes


def process_pose_predictions(predictions):
    labels = load_labels('models/positions.TensorFlow/labels.txt')
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = float(predictions[0][predicted_index])
    return {'label': predicted_label, 'confidence': confidence}


def process_detection_results(boxes, scores, classes, image_width, image_height, labels_path):
    labels = load_labels(labels_path)
    results = []
    num_detections = len(scores)
    for i in range(num_detections):
        confidence = scores[i]
        if confidence >= 0.5:
            class_id = int(classes[i])
            label = labels[class_id]
            box = boxes[i]
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)
            results.append({
                'label': label,
                'confidence': float(confidence),
                'bbox': [xmin, ymin, xmax, ymax]
            })
    return results


def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def analyze_frame(frame_path, pose_graph, watermark_graph, genital_graph, penetration_graph):
    # Load the original image to get width and height for later use
    image = cv2.imread(frame_path)
    image_height, image_width, _ = image.shape

    # Preprocess image for Pose Classification model
    pose_input_size = (300, 300)  # Adjust based on your classification model's expected input size
    pose_image_np = preprocess_image(frame_path, pose_input_size)

    # Preprocess image for Detection models
    detection_input_size = (320, 320)  # Adjust based on your detection models' expected input size
    detection_image_np = preprocess_image(frame_path, detection_input_size)

    # Run inference for each model
    # Pose Classification
    pose_predictions = run_classification_inference(pose_graph, pose_image_np)
    pose_result = process_pose_predictions(pose_predictions)

    # Watermark Detection
    wm_boxes, wm_scores, wm_classes = run_object_detection_inference(watermark_graph, detection_image_np)
    watermark_results = process_detection_results(
        wm_boxes, wm_scores, wm_classes,
        image_width, image_height,
        'models/watermark.TensorFlow/labels.txt'
    )

    # Genital Detection
    genital_boxes, genital_scores, genital_classes = run_object_detection_inference(genital_graph, detection_image_np)
    genital_results = process_detection_results(
        genital_boxes, genital_scores, genital_classes,
        image_width, image_height,
        'models/genitals.TensorFlow/labels.txt'
    )

    # Penetration Detection
    penetration_boxes, penetration_scores, penetration_classes = run_object_detection_inference(penetration_graph,
                                                                                                detection_image_np)
    penetration_results = process_detection_results(
        penetration_boxes, penetration_scores, penetration_classes,
        image_width, image_height,
        'models/penetration.TensorFlow/labels.txt'
    )

    # Combine results
    analysis_results = {
        'pose': pose_result,
        'watermarks': watermark_results,
        'genitals': genital_results,
        'penetrations': penetration_results
    }

    # Save results
    output_dir = os.path.join('data', 'images')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(frame_path)
    result_path = os.path.join(output_dir, base_name)
    cv2.imwrite(result_path, image)  # Save the original image

    # Save analysis results as JSON
    json_path = os.path.splitext(result_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)


def main():
    # Load models once
    try:
        logger.info('Loading models...')
        pose_graph = load_frozen_graph('models/positions.TensorFlow/model.pb')
        watermark_graph = load_frozen_graph('models/watermark.TensorFlow/model.pb')
        genital_graph = load_frozen_graph('models/genitals.TensorFlow/model.pb')
        penetration_graph = load_frozen_graph('models/penetration.TensorFlow/model.pb')
        logger.info('All models loaded successfully.')
    except Exception as e:
        logger.error('Error loading models.', exc_info=True)
        return

    video_dir = os.path.join('data', 'clips')
    video_files = [os.path.join(root, file)
                   for root, _, files in os.walk(video_dir)
                   for file in files if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.ts'))]
    total_videos = len(video_files)
    logger.info(f'Found {total_videos} video(s) to process.')

    for idx, video_path in enumerate(video_files, start=1):
        logger.info(f'Processing video {idx}/{total_videos}: {video_path}')
        try:
            frame_path = extract_sharpest_frame(video_path)
            if frame_path:
                analyze_frame(
                    frame_path,
                    pose_graph,
                    watermark_graph,
                    genital_graph,
                    penetration_graph
                )
                logger.info(f'Analysis complete for video: {video_path}')
            else:
                logger.warning(f'No frame extracted from video: {video_path}')
        except Exception as e:
            logger.error(f'Error processing video {video_path}', exc_info=True)


if __name__ == '__main__':
    main()
