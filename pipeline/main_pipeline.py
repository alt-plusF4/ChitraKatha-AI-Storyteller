import os
import time
import uuid
import shutil
from moviepy.editor import AudioFileClip

from pipeline import (
    module0_casting, 
    module1_screenwriting, 
    module2_voiceover, 
    module3_image_generation, 
    module4_postproduction
)

def create_story_video(prompt: str, language: str = "English", tone: str = "Default"):
    """
    Main pipeline for generating an AI animated story with audio, images, and video.
    Returns the path to the final video and the full story text.
    """
    print("\n--- Starting New Story Generation ---")
    pipeline_start = time.time()

    # --- Step 0: Unique folder per request ---
    request_id = uuid.uuid4().hex[:8]
    project_name = f"generated_story_{request_id}"
    os.makedirs(project_name, exist_ok=True)

    temp_folder = os.path.join(project_name, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    temp_audio_files = []
    temp_image_files = []

    # --- Step 1: Enrich Prompt and Create Cast List ---
    rich_concept, cast_list = module0_casting.enrich_prompt(simple_prompt=prompt, story_tone=tone)

    # --- Step 2: Generate Story Text and Scene List ---
    story_text = module1_screenwriting.generate_story_text(rich_concept, cast_list)
    scene_list = module1_screenwriting.generate_scene_list_from_story(story_text, cast_list, rich_concept)

    if not scene_list:
        return None, "Failed to generate story content. Please try a different prompt."

    video_clips = []
    full_story_text = " ".join([scene.get("sentence", "") for scene in scene_list])

    # --- Step 3: Loop Through Scenes ---
    for i, scene_data in enumerate(scene_list):
        sentence = scene_data.get("sentence")
        updated_cast = scene_data.get("characters", [])
        shot = scene_data.get("shot", [])
        shot_type = scene_data.get("shot_type", [])

        if not sentence or not shot or not updated_cast:
            print(f"Scene {i+1} missing critical data. Skipping.")
            continue

        print(f"\n--- Processing Scene {i+1}/{len(scene_list)}: {sentence} ---")

        # --- Generate Scene Audio ---
        audio_path = os.path.join(temp_folder, f"scene_{i+1:02d}.mp3")
        audio_success = module2_voiceover.generate_audio(
            text=sentence, lang=language, filename=audio_path, story_tone=tone
        )
        if not audio_success:
            print(f"Audio generation failed. Skipping scene.")
            continue
        temp_audio_files.append(audio_path)

        if isinstance(shot, str):
            shot = [shot]

        with AudioFileClip(audio_path) as main_audio_clip:
            audio_duration = main_audio_clip.duration
            duration_per_shot = audio_duration / len(shot)

            for j, visual_prompt in enumerate(shot):
                shot_path_base = os.path.join(temp_folder, f"scene_{i+1:02d}_shot_{j+1:02d}")
                image_path = f"{shot_path_base}.png"
                sub_audio_path = f"{shot_path_base}.mp3"

                # --- Generate Image ---
                module3_image_generation.generate_image(
                    shot_type=shot_type,
                    visual_prompt=visual_prompt,
                    updated_cast=updated_cast,
                    filename=image_path
                )
                temp_image_files.append(image_path)

                # --- Split Audio ---
                sub_audio_clip = main_audio_clip.subclip(j * duration_per_shot, (j + 1) * duration_per_shot)
                sub_audio_clip.write_audiofile(sub_audio_path, logger=None)
                temp_audio_files.append(sub_audio_path)

                # --- Create Scene Clip ---
                final_audio_clip = AudioFileClip(sub_audio_path)
                scene_clip = module4_postproduction.create_scene_clip(
                    image_path=image_path,
                    audio_clip=final_audio_clip,
                    subtitle_text=sentence
                )
                video_clips.append(scene_clip)

    if not video_clips:
        return None, "Video generation failed. No valid scenes created."

    # --- Step 4: Assemble Final Video ---
    final_video_path = os.path.join(project_name, "final_story.mp4")
    module4_postproduction.assemble_video(video_clips, output_filename=final_video_path)

    # --- Optional: Clean up temp files ---
    shutil.rmtree(temp_folder)

    pipeline_end = time.time()
    print(f"\n[Main] Pipeline complete in {(pipeline_end - pipeline_start)/60:.2f} minutes.")

    return final_video_path, full_story_text
