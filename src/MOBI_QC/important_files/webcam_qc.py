#%%
import cv2
import pyxdf
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

#%%


def count_faces_in_video(video_path:str, cam_df:pd.DataFrame, stim_df:pd.DataFrame, task:str, frame_skip=10):
    """Counts the number of frames with detected faces using a deep learning-based face detector,
    and returns the indices of the frames where faces are detected."""
    
    # frames of interest
    foi = get_event_data(task, df=cam_df, stim_df=stim_df)['frame_num'].values


    # Load the pre-trained MobileNet SSD model and config file for face detection
    net = cv2.dnn.readNetFromCaffe(
        '../important_files/deploy.prototxt',  # Path to the model configuration file
        '../important_files/res10_300x300_ssd_iter_140000_fp16.caffemodel'  # Path to the model weights
    )

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return 0, []

    face_count = 0
    frame_idx = 0
    frames_with_faces = []
    frames_checked = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Skip frames for efficiency
        if frame_idx % frame_skip == 0:
            # Check if the current frame index is in the list of frames of interest
            if frame_idx in foi:
                frames_checked.append(frame_idx)
                # Prepare the frame for the DNN model (resize to 300x300 and normalize)
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
                net.setInput(blob)

                # Perform face detection
                detections = net.forward()

                # Check if a face is detected
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Confidence threshold
                        face_count += 1
                        frames_with_faces.append(frame_idx)  # Save the frame index where a face is detected
                        break  # Count one face per frame

        frame_idx += 1  # Increment frame counter

    cap.release()
    return face_count, frames_with_faces, frames_checked


def extract_frames(video_path, frame_indices, resize_scale=0.5):
    """Extracts specific frames from a video and returns them as a list of images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to specific frame
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Convert BGR to RGB (OpenCV loads images in BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame if needed
        if resize_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LANCZOS4)

        frames.append(frame)

    cap.release()
    return frames

def plot_frames_with_wrap(frames, highlight_indices=[], overlap_ratio=0.3, frames_per_row=20):
    """Plots frames with 30% overlap, wrapping to a new row every `frames_per_row` frames."""
    if not frames:
        print("No frames to display.")
        return
    
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    overlap_pixels = int(frame_width * overlap_ratio)  # Amount of overlap in pixels
    num_rows = int(np.ceil(len(frames) / frames_per_row))  # Determine number of rows

    # Compute canvas size
    row_width = frame_width + (frames_per_row - 1) * (frame_width - overlap_pixels)  # Width for each row
    total_height = num_rows * frame_height  # Stack rows vertically

    canvas = np.ones((total_height, row_width, 3), dtype=np.uint8) * 255  # White background

    for i, frame in enumerate(frames):
        row = i // frames_per_row  # Determine which row the frame belongs to
        col = i % frames_per_row   # Determine position in the row

        x_start = col * (frame_width - overlap_pixels)  # X position
        y_start = row * frame_height  # Y position (new row starts here)

        # If frame should be highlighted, make it red
        if i in highlight_indices:
            red_frame = frame.copy()
            red_frame[:, :, 1:] = 0  # Set green and blue channels to 0, keeping only red
            frame = red_frame

        # Apply random noise to the frame
        noise = np.random.normal(0, 1, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        # Overlay the frame
        canvas[y_start:y_start + frame_height, x_start:x_start + frame_width] = frame

    # Plot the final stitched image
    plt.figure(figsize=(frames_per_row * 1.5, num_rows * 2.5))  # Adjust figure size dynamically
    plt.imshow(canvas)
    plt.axis("off")
    plt.show()
    return canvas



def webcam_qc(xdf_filename:str, video_file:str, stim_df:pd.DataFrame, task:str):
    cam_df = import_video_data(xdf_filename)
    
    sub_id = xdf_filename.split('sub-')[1].split('/')[0]


    exp_start = stim_df.loc[stim_df.event == f'Onset_{task}', 'lsl_time_stamp'].values[0]
    exp_end = stim_df.loc[stim_df.event == f'Offset_{task}', 'lsl_time_stamp'].values[0]
    experiment_dur = exp_end - exp_start

    cam_exp = get_event_data(task, cam_df, stim_df=stim_df)

    start = cam_exp['lsl_time_stamp'].values[0]
    stop = cam_exp['lsl_time_stamp'].values[-1]
    cam_dur = stop - start
    vars = {}
    
    if abs(experiment_dur - cam_dur) < 0.1:
        print(f'{task} duration matches camera duration!')
        vars[f'collected_full_{task}'] = True
        print(f'{task}: ', experiment_dur)
        print('Webcam Stream: ', cam_dur)
    else:
        print(f'{task} duration does not match camera duration!')
        vars[f'collected_full_{task}'] = True

        print(f'{task}: ', experiment_dur)
        print('Webcam Stream: ', cam_dur)

    sampling_rate = 1/cam_df.frame_time_sec.diff().mean()  # 30 fps
    vars['sampling_rate'] = sampling_rate
    cam_df['face_detected'] = False

    fc, face_frames, frames_checked = count_faces_in_video(video_file, cam_df=cam_df, stim_df=stim_df, task=task, frame_skip=10)
    frames_without_faces = [frame for frame in frames_checked if frame not in face_frames]
    face_perc = fc/len(frames_checked)
    vars['face_perc'] = face_perc


    frame_indices = frames_checked[5::5] # plot every 5th frame
    highlight_indices = [frame_indices.index(xx)  for xx in [x for x in frames_without_faces if x in frame_indices]] # Indices of frames to be highlighted in red
    frames = extract_frames(video_file, frame_indices, resize_scale=0.35)
    canvas = plot_frames_with_wrap(frames, highlight_indices=highlight_indices, overlap_ratio=0.3, frames_per_row=30)
    # same the canvas
    plt.imsave(f'report_images/{sub_id}_webcam_qc.png', canvas)
    return vars, cam_df


# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass



