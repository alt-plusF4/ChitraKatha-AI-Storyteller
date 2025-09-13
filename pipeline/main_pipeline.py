#(Main Pipeline)

import os
import re
import time
from dotenv import load_dotenv
from moviepy.editor import AudioFileClip

from pipeline import module0_casting, module1_screenwriting, module2_voiceover, module3_image_generation, module4_postproduction

def create_story_video(prompt: str, language: str = "English", tone: str = "Default"):
    """
    The main, fully functional pipeline that uses the professional "Director" AI workflow.
    """
    print("\n--- Starting New Story Generation ---")
    pipeline_start = time.time()
    project_name = "generated_story"
    os.makedirs(project_name, exist_ok=True)
    
    temp_audio_files = [] # Keep track of temporary files to clean up later

    # --- Step 1: Enrich Prompt and Create the Initial Cast List ---
    rich_concept, cast_list = enrich_prompt(simple_prompt=prompt, story_tone=tone)

    # --- Step 2: Generate the Full Scene-by-Scene Breakdown ---
    story_text = generate_story_text(rich_concept, cast_list)
    scene_list = generate_scene_list_from_story(story_text, cast_list, rich_concept)

    if not scene_list:
        return None, "Failed to generate story content. Please try a different prompt."

    video_clips = []
    full_story_text = " ".join([scene.get("sentence", "") for scene in scene_list])

    # --- Step 3: Loop Through Each Scene to Generate Media ---
    for i, scene_data in enumerate(scene_list):
        sentence = scene_data.get("sentence")
        updated_cast = scene_data.get("characters", [])
        shot = scene_data.get("shot", [])
        shot_type = scene_data.get("shot_type", [])

        if not sentence or not shot or not updated_cast:
            print(f"Scene {i+1} is missing critical data. Skipping.")
            continue

        print(f"\n--- Processing Scene {i+1}/{len(scene_list)}: {sentence} ---")

        audio_path = os.path.join(project_name, f"scene_{i+1:02d}.mp3")
        audio_success = generate_audio(text=sentence, lang=language, filename=audio_path, story_tone=tone)
        if not audio_success:
            print(f"Audio generation failed. Skipping scene.")
            continue

        if isinstance(shot, str):
            shot = [shot]
        
        with AudioFileClip(audio_path) as main_audio_clip:        
            main_audio_clip = AudioFileClip(audio_path)
            audio_duration = main_audio_clip.duration
            duration_per_shot = audio_duration / len(shot)

            for j, visual_prompt in enumerate(shot):
                shot_path_base = os.path.join(project_name, f"scene_{i+1:02d}_shot_{j+1:02d}")
                image_path = f"{shot_path_base}.png"

                generate_image(
                    shot_type = shot_type,
                    visual_prompt=visual_prompt,
                    updated_cast=updated_cast,
                    filename=image_path
                )

                start_time = j * duration_per_shot
                sub_audio_clip_obj = main_audio_clip.subclip(start_time, start_time + duration_per_shot)
                
                # --- CORRECTED: Save subclip to a file and reload it without a 'with' statement ---
                sub_audio_path = f"{shot_path_base}.mp3"
                sub_audio_clip_obj.write_audiofile(sub_audio_path, logger=None)
                temp_audio_files.append(sub_audio_path) # Add to cleanup list
                
                # Use a new, clean AudioFileClip object that will stay open
                final_audio_clip = AudioFileClip(sub_audio_path)
                
                scene_clip = create_scene_clip(
                    image_path=image_path,
                    audio_clip=final_audio_clip,
                    subtitle_text=sentence
                )
                video_clips.append(scene_clip)

    if not video_clips:
        return None, "Video generation failed because no valid scenes could be created."

    # --- Step 4: Assemble the Final Video ---
    final_video_path = os.path.join(project_name, "final_story.mp4")
    assemble_video(video_clips, output_filename=final_video_path)

    pipeline_end = time.time()
    print(f"\n[Main] Pipeline complete in {(pipeline_end - pipeline_start) / 60:.2f} minutes.")

    return final_video_path, full_story_text
