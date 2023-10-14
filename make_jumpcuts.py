from moviepy.editor import VideoFileClip, concatenate_videoclips

def jump_between_videos(filepaths, output_path):
    clips = [VideoFileClip(fp) for fp in filepaths]
    
    # Determine the smallest resolution among the videos
    min_width = min(clip.size[0] for clip in clips)
    min_height = min(clip.size[1] for clip in clips)
    
    # Resize and/or crop each clip to the smallest resolution
    resized_clips = []
    for clip in clips:
        # Center-crop to the target resolution
        cropped_clip = clip.crop(
            x_center=clip.size[0]/2, y_center=clip.size[1]/2,
            width=min_width, height=min_height
        )
        resized_clips.append(cropped_clip)
    
    # Extract one frame from each video (assumes 30fps, so 1/30 duration)
    subclips = [clip.subclip(t, t+1/30) for t, clip in enumerate(resized_clips)]
    
    # Concatenate until 2 seconds long or out of frames
    n_clips_needed = int(2 * 30)  # 2 seconds * 30fps
    final_clips = (subclips * (n_clips_needed // len(subclips)))[:n_clips_needed]
    
    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video.write_videofile(output_path, fps=30)

# Test
filepaths = ['inputs/Fallon_Kimmel_Demo.mp4', 'inputs/Friends_Clip.mp4', 'inputs/noface_demo.mp4', \
             'inputs/Smiling_Control_Video.mp4', 'inputs/Verification_Demo.mp4']
output_path_now = 'inputs/jumpcuts.mp4'
jump_between_videos(filepaths, output_path_now)
