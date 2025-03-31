import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
import cv2
from tqdm import tqdm
import dlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import threading
import tensorflow as tf
import argparse

# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Path constants
PDD_PATH = '~/scratchDirectory/PDD'
OUTPUT_DIR = 'data'

# Helper Functions
def extract_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    return total_frames, fps, duration

def extract_10_frames(video_path, fps, start_time, end_time, save_dir=None, segment_idx=None):
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    
    # Calculate start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Extract all frames in the time range
    for frame_id in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            all_frames.append(frame)
    
    cap.release()
    
    # Calculate segment size (number of frames per segment)
    total_frames = len(all_frames)
    segment_size = total_frames // 5
    
    # Sample 2 frames from each segment
    sampled_frames = []
    for i in range(5):
        segment_start = i * segment_size
        segment_end = (i + 1) * segment_size
        
        # Get 2 frames from the middle of each segment
        mid_point = (segment_start + segment_end) // 2
        if mid_point < len(all_frames) and mid_point + 1 < len(all_frames):
            sampled_frames.append(all_frames[mid_point])
            sampled_frames.append(all_frames[mid_point + 1])
            
            # Save frames if directory is provided
            if save_dir and segment_idx is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                segment_name = f"segment_{segment_idx}"
                cv2.imwrite(os.path.join(save_dir, f"{segment_name}_frame_{i*2}.jpg"), all_frames[mid_point])
                cv2.imwrite(os.path.join(save_dir, f"{segment_name}_frame_{i*2+1}.jpg"), all_frames[mid_point + 1])
    
    return sampled_frames

# Flicker Detection
def detect_flicker(frames):
    flow_magnitudes = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        next_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Compute optical flow between prev and current
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.mean(magnitude))

        prev_gray = next_gray
    flow_magnitudes = StandardScaler().fit_transform(np.array(flow_magnitudes).reshape(-1, 1)).flatten().tolist()
    return flow_magnitudes

# Lip Movement
def get_lip_movement(frames):
    """
    Calculates the variance of lip height movement from a list of frames.
    - Assumes frames are in BGR format.
    - Detects the largest face in each frame.
    - Extracts top and bottom lip landmarks.
    - Returns variance of lip height movement as a single float.
    """
    movements = []

    try:
        for frame in frames:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                if len(faces) == 0:
                    continue

                face = sorted(faces, key=lambda f: f.width() * f.height(), reverse=True)[0]
                landmarks = predictor(gray, face)

                lip_height = landmarks.part(66).y - landmarks.part(62).y
                movements.append(lip_height)

            except Exception as inner_e:
                print(f"Error processing frame: {inner_e}")
                continue

        if len(movements) < 2:
            return [0.0]
        movements = StandardScaler().fit_transform(np.array(movements).reshape(-1, 1)).flatten().tolist()
        result = float(np.mean(np.abs(np.diff(movements))))
        return [result]

    except Exception as e:
        print(f"Unexpected error in get_lip_movement: {e}")
        return [0.0]

# Blink Detection
def detect_blinks(frames):
    """
    Detects the number of blinks from a list of frames.
    Uses a simple eye-aspect-ratio (EAR) based difference between eye heights.
    """
    blink_count = 0

    for frame in frames:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = np.mean([landmarks.part(i).y for i in range(36, 42)])
                right_eye = np.mean([landmarks.part(i).y for i in range(42, 48)])

                eyes_aspect_ratio = abs(left_eye - right_eye)
                if eyes_aspect_ratio < 1.2:  # Simple threshold — tune as needed
                    blink_count += 1

        except Exception as e:
            print(f"Error processing frame in blink detection: {e}")
            continue

    return [blink_count]

# Head Movement Anomalies
def extract_head_pose(frames):
    """
    Estimates vertical head movement using the distance between nose tip and chin
    from a list of frames. Always returns a list of 10 normalised values.
    If fewer than 10 values are available, pads with 0.0.
    If more than 10, trims to 10.
    """
    movements = []

    for frame in frames:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) == 0:
                movements.append(0.0)
                continue

            face = sorted(faces, key=lambda f: f.width() * f.height(), reverse=True)[0]
            landmarks = predictor(gray, face)

            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            distance = float(np.linalg.norm(np.array(nose_tip) - np.array(chin)))
            movements.append(distance)

        except Exception as e:
            print(f"Error processing frame in head pose extraction: {e}")
            movements.append(0.0)

    # Scale only if we have non-zero values
    if any(movements):
        movements = StandardScaler().fit_transform(np.array(movements).reshape(-1, 1)).flatten().tolist()

    # Adjust to exactly 10 elements
    if len(movements) < 10:
        movements += [0.0] * (10 - len(movements))
    elif len(movements) > 10:
        movements = movements[:10]

    return movements

# Pulse Detection
def detect_pulse(frames):
    """
    Extracts pulse signal from a list of frames by averaging the green channel
    in a fixed forehead region. Applies StandardScaler to the signal.
    Returns a normalised NumPy array of pulse signal.
    """
    pulse_signals = []

    for frame in frames:
        try:
            # Ensure the frame is large enough
            if frame.shape[0] > 150 and frame.shape[1] > 150:
                roi = frame[100:150, 100:150, 1]  # green channel from forehead region
                avg_green = np.mean(roi)
                pulse_signals.append(avg_green)
            else:
                # If frame is too small, use what we can
                roi = frame[:min(frame.shape[0], 50), :min(frame.shape[1], 50), 1]
                avg_green = np.mean(roi)
                pulse_signals.append(avg_green)
        except Exception as e:
            print(f"Error processing frame in pulse detection: {e}")
            continue

    # Apply standardization only if we have data
    if pulse_signals:
        pulse_signals = StandardScaler().fit_transform(np.array(pulse_signals).reshape(-1, 1)).flatten().tolist()

    return pulse_signals

# PSNR and SSIM
def compute_ssim_psnr(frames):
    psnr_vals = []
    ssim_vals = []

    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return 100  # identical
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    for i in range(len(frames) - 1):
        img1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        psnr_val = calculate_psnr(img1, img2)
        ssim_val, _ = ssim(img1, img2, full=True)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
    
    # Standardize the values
    if psnr_vals:
        psnr_vals = StandardScaler().fit_transform(np.array(psnr_vals).reshape(-1, 1)).flatten().tolist()
    
    if ssim_vals:
        ssim_vals = StandardScaler().fit_transform(np.array(ssim_vals).reshape(-1, 1)).flatten().tolist()

    return psnr_vals, ssim_vals

# TFRecord functions
def _bytes_feature(value):
    """Convert a string / byte to a TFRecord bytes feature."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Convert float values to a TensorFlow feature."""
    if isinstance(value, str):
        # Convert string representation of list to numpy array
        value = np.array(eval(value))
    elif not isinstance(value, np.ndarray):
        value = np.array([value])
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def _int_feature(value):
    """Convert an integer value to a TensorFlow feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_features(row, index):
    """Convert a row of features into a TFRecord format."""
    feature_dict = {
        'filename': _bytes_feature(str(index).encode('utf-8')),
        'fake': _int_feature(row['fake']),
        'flicker': _float_feature(row['flicker']),
        'lip_movement_variance': _float_feature(row['lip_movement_variance']),
        'blink': _float_feature(row['blink']),
        'head_movement': _float_feature(row['head_movement']),
        'pulse': _float_feature(row['pulse']),
        'psnr': _float_feature(row['psnr']),
        'ssim': _float_feature(row['ssim']),
        'feature_lengths': _float_feature(row['feature_lengths'])
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString()

def write_tfrecord(df_features, output_file="pdd_features.tfrecord"):
    """Write DataFrame features to a TFRecord file."""
    with tf.io.TFRecordWriter(output_file) as writer:
        for idx, row in tqdm(df_features.iterrows(), total=len(df_features), desc="Writing TFRecord"):
            example = serialize_features(row, idx)
            writer.write(example)
    
    print(f"TFRecord saved: {output_file}")

def read_tfrecord(tfrecord_path):
    """Read TFRecord file and convert to DataFrame."""
    
    # Define feature description for parsing
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'fake': tf.io.FixedLenFeature([], tf.int64),
        'flicker': tf.io.VarLenFeature(tf.float32),
        'lip_movement_variance': tf.io.VarLenFeature(tf.float32),
        'blink': tf.io.VarLenFeature(tf.float32),
        'head_movement': tf.io.VarLenFeature(tf.float32),
        'pulse': tf.io.VarLenFeature(tf.float32),
        'psnr': tf.io.VarLenFeature(tf.float32),
        'ssim': tf.io.VarLenFeature(tf.float32),
        'feature_lengths': tf.io.VarLenFeature(tf.float32)
    }

    def _parse_function(example_proto):
        """Parse TFRecord example."""
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert sparse tensors to dense and then to numpy arrays
        features = {}
        for key in parsed_features:
            if key == 'filename':
                features[key] = parsed_features[key].numpy().decode('utf-8')
            elif key == 'fake':
                features[key] = parsed_features[key].numpy()
            else:
                features[key] = tf.sparse.to_dense(parsed_features[key]).numpy()
        
        return features

    # Read TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse all examples
    data = []
    for raw_record in dataset:
        features = _parse_function(raw_record)
        data.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.set_index('filename', inplace=True)
    
    return df

# Main Processing Function
def process_video(filename):
    if not filename.lower().endswith('.mp4'):
        return None

    video_path = os.path.join(PDD_PATH, filename)
    
    # Create output directory for frames
    video_name = os.path.splitext(filename)[0]
    output_frames_dir = os.path.join(OUTPUT_DIR, video_name)
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    try:
        tqdm.write(f"Processing {filename} on thread {threading.current_thread().name}")
        total_frames, fps, duration = extract_video_info(video_path)
        segment_timestamps = [(i * (duration / 5), (i + 1) * (duration / 5)) for i in range(5)]
        feature_lengths = []
        flicker_vals = []
        lip_var = []
        blink_feature = []
        head_movement = []
        pulse_vals = []
        psnr_vals = []
        ssim_vals = []

        for i, (start_time, end_time) in enumerate(segment_timestamps):
            # Extract frames and save them
            frames = extract_10_frames(
                video_path, fps, start_time, end_time, 
                save_dir=output_frames_dir, segment_idx=i
            )

            flicker_vals += detect_flicker(frames)
            lip_var += get_lip_movement(frames)
            blink_feature += detect_blinks(frames)
            head_movement += extract_head_pose(frames)
            pulse_vals += detect_pulse(frames)
            psnr_val, ssim_val = compute_ssim_psnr(frames)
            psnr_vals += psnr_val
            ssim_vals += ssim_val

        feature_lengths.append(len(flicker_vals))
        feature_lengths.append(len(lip_var))
        feature_lengths.append(len(blink_feature))
        feature_lengths.append(len(head_movement))
        feature_lengths.append(len(pulse_vals))
        feature_lengths += [len(psnr_vals), len(ssim_vals)]
        
        # Determine if this is a fake video based on filename
        if int(filename[2:4]) < 8:
            label = 1
        else:
            label = 0
            
        return {
            'filename': filename,
            'fake': label,
            "flicker": flicker_vals,
            "lip_movement_variance": lip_var,
            "blink": blink_feature,
            "head_movement": head_movement,
            "pulse": pulse_vals,
            "psnr": psnr_vals,
            "ssim": ssim_vals,
            'feature_lengths': feature_lengths
        }

    except Exception as e:
        tqdm.write(f"Error processing {filename}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDD dataset videos and extract features')
    parser.add_argument('--pdd_path', type=str, default='~/scratchDirectory/PDD', help='Path to PDD dataset')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save extracted frames')
    parser.add_argument('--output_tfrecord', type=str, default='pdd_features.tfrecord', help='Output TFRecord file path')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads to use')
    args = parser.parse_args()
    
    # Update global paths
    global PDD_PATH, OUTPUT_DIR
    PDD_PATH = args.pdd_path
    OUTPUT_DIR = args.output_dir
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get video files
    filenames = os.listdir(PDD_PATH)
    
    # Process videos
    data = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_video, fname) for fname in filenames if fname.lower().endswith('.mp4')]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            if result:
                data.append(result)
    
    # Convert to DataFrame
    df_features = pd.DataFrame(data)
    df_features.set_index('filename', inplace=True)
    
    # Write to TFRecord
    write_tfrecord(df_features, args.output_tfrecord)
    
    print(f"Processed {len(df_features)} videos")
    print(f"Extracted frames saved to {OUTPUT_DIR}")
    print(f"Features saved to {args.output_tfrecord}")

if __name__ == "__main__":
    main() 