import os
import yt_dlp

# Ensure output directory exists
if not os.path.exists('./video'):
    os.makedirs('./video')

# Define options for downloading
options = {
    'format': 'bestvideo+bestaudio/best',  # Best video and audio quality
    'outtmpl': './video/%(title)s.%(ext)s',  # Save files to ./video/ with video title as filename
    'noplaylist': True,  # Download only the video, not the playlist
    'postprocessors': [
        {
            'key': 'FFmpegVideoConvertor',  # Use FFmpeg for conversion
            'preferedformat': 'mp4',  # Correct spelling for 'preferredformat'
        }
    ],
}

# Function to download a single video
def download_video(url):
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")

# Function to process URLs from a text file
def process_urls_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as file:
        for line in file:
            url = line.strip()  # Remove whitespace and newline characters
            if url:
                print(f"Downloading: {url}")
                download_video(url)

# Example usage
if __name__ == "__main__":
    file_path = "video_sources.txt"  # Replace with the path to your text file
    process_urls_from_file(file_path)
