import os
import cv2
import shutil

def extract_frames_from_videos(video_dir, output_dir, frequency=3):
    """
    Extracts frames from videos at a specified frequency and saves them to an output directory.

    Args:
        video_dir (str): Directory containing the video files.
        output_dir (str): Directory where the extracted frames will be saved.
        frequency (int): The time interval (in seconds) at which frames will be saved.
    """
    # Remove the directory if it exists, then create a new one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the directory and its contents
    
    # Create the new output directory
    os.makedirs(output_dir)

    # Iterate through all files in the video directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            
            # Capture the video using OpenCV
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_length = total_frames / fps  # Length of video in seconds

            # Save a frame every 'frequency' seconds
            frame_interval = frequency  # seconds
            frame_count = 0
            success = True
            
            while success:
                frame_position = int(frame_interval * fps * frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                success, frame = cap.read()
                
                if success:
                    # Save the frame as an image
                    output_frame_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.jpg")
                    cv2.imwrite(output_frame_path, frame)
                    frame_count += 1
                
                if frame_position >= total_frames:
                    break
            
            # Release the video capture object
            cap.release()

    print("Video processing completed!")

if __name__ == "__main__" :
    extract_frames_from_videos("video", "frames", 3)
