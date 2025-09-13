import os
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, concatenate_audioclips, TextClip, vfx
)
from PIL import Image
import moviepy.audio.fx.all as afx
import numpy as np
from pipeline.config import GOOGLE_API_KEY, ELEVEN_API_KEY, HF_API_TOKEN

def create_scene_clip(image_path: str, audio_clip: AudioFileClip, subtitle_text: str):
    """
    Creates a professional-quality video scene with smooth, slow Ken Burns effect
    (stable zoom + slow horizontal pan) and Netflix-style subtitles.
    """
    duration = audio_clip.duration

    img_clip = ImageClip(image_path).resize(height=1080).set_duration(duration)
    w, h = img_clip.size

    # --- Stable Ken Burns Effect: zoom + left-to-right pan ---
    zoom_factor = 1.15
    pan_start = -0.02
    pan_end = 0.02

    # Apply smooth zoom using moviepy's built-in, optimized vfx
    animated_clip = img_clip.fx(vfx.resize, lambda t: 1 + (zoom_factor - 1) * (t / duration))
    
    # Apply slow left-to-right pan using moviepy's built-in scroll effect
    animated_clip = animated_clip.fx(
        vfx.scroll,
        w=animated_clip.w, # Use the zoomed width
        h=animated_clip.h, # Use the zoomed height
        x_speed=(pan_end - pan_start) * w / duration,
        y_speed=0
    )
    
    # Crop the animated clip back to the original size
    animated_clip = animated_clip.crop(x_center=animated_clip.w/2, y_center=animated_clip.h/2, width=w, height=h)


    # --- Subtitles with automatic wrapping and padding ---
    max_subtitle_width = int(w * 0.80) 
    font_size = 24

    subtitle_clip = TextClip(
        subtitle_text,
        fontsize=font_size,
        color='white',
        font='Arial',
        method='caption',
        size=(max_subtitle_width, None),  # Width fixed, height auto
        bg_color='rgba(0,0,0,0.5)',
        align='center'
    ).set_position(('center', 0.85), relative=True).set_duration(duration)

    # Composite clip
    final_clip = CompositeVideoClip([animated_clip, subtitle_clip])
    final_clip.audio = audio_clip

    return final_clip


def assemble_video(clips: list, output_filename="final_story.mp4", crossfade_duration=0.5):

    if not clips:
        raise ValueError("No clips were provided for video assembly.")

    print("Assembling final animated video...")

    faded_clips = [clips[0]]
    for clip in clips[1:]:
        faded_clips.append(clip.crossfadein(crossfade_duration))
        
    final_video = concatenate_videoclips(faded_clips, padding=-crossfade_duration, method="compose")
    
    audio_clips = [c.audio for c in clips]

    final_audio = concatenate_audioclips(audio_clips).fx(afx.audio_fadeout, crossfade_duration)
    
    # 4. Set the final audio to the final video.
    final_video.audio = final_audio

    # GPU-accelerated / multi-threaded encoding
    final_video.write_videofile(
        output_filename,
        fps=24,
        codec="libx264",           
        audio_codec="aac",
        threads=os.cpu_count(),
        preset="fast",
        ffmpeg_params=["-crf", "20"],
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
