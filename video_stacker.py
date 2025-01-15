from moviepy.editor import VideoFileClip, clips_array

# Load your video files
video1 = VideoFileClip("data/videos/test/live_plot.mp4")
video2 = VideoFileClip("data/videos/test/robot_walking.mp4")

# Ensure both videos have the same height
if video1.h != video2.h:
    max_height = max(video1.h, video2.h)
    video1 = video1.resize(height=max_height)
    video2 = video2.resize(height=max_height)

# Combine videos side by side
final_video = clips_array([[video1, video2]])

# Write the output to a file
final_video.write_videofile("output_video.mp4", codec="libx264")
