# module_3_image_generator.py
import os
import io
import time
import requests
from PIL import Image
from typing import List, Dict, Any
from dotenv import load_dotenv
from pipeline.config import GOOGLE_API_KEY, ELEVEN_API_KEY, HF_API_TOKEN

# --- Environment Setup ---
load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def generate_image(visual_prompt: str, updated_cast: List[Dict[str, Any]], filename: str, shot_type: str):
    # --- Step 1: Build a consistent character block ---
    character_prompts = []
    seeds = []
    AGE_STAGE_TO_MODIFIER = {
        "child": "round face, smooth skin, wide eyes, clean shave, short body, black hair, age ~5",
        "youth": "soft features, smooth skin, clean shave, medium tall body, bright expressive eyes, slim energetic posture, black hair, age ~15",
        "adult": "defined jawline, mature balanced features, clean shave, tall body, steady eyes, black hair, confident posture, age ~25",
        "elder": "wrinkled skin, deep-set eyes, gray hair, tall body with stooped dignified posture, age ~65"
    }

    for char in updated_cast:
        name = char.get("name", "a character")
        identity = char.get("identity_tag", name.lower().replace(" ", "_") + "_canonical")
        anchors = ", ".join(char.get("visual_anchors", []))
        face_anchors = ", ".join(char.get("face_anchors", []))
        age_stage = char.get("age_stage", "adult")
        age_mod = AGE_STAGE_TO_MODIFIER.get(age_stage.lower(), "mature features")
        seed = char.get("seed")

        if seed is not None:
            seeds.append(seed)

        character_prompts.append(
            f"Character '{name}' (identity: {identity}, seed: {seed}) → "
            f"Immutable visual anchor: {anchors}, Immutable face anchor: {face_anchors} . Depict dynamically with {age_mod}."
        )
    
    character_description = " | ".join(character_prompts)

    # --- Step 2: Style + Shot prompt ---
    style_prompt = (
        "children's book illustration, claymation style, soft textures, "
        "cinematic lighting, whimsical, heartwarming, detailed Indian setting"
    )
    full_prompt = (
        f"Shot type: {shot_type}. "
        f"Style: {style_prompt}. "
        f"Scene showing: {character_description}. "
        f"Scene Description: {visual_prompt}. "
    )

    # --- Step 3: Negative prompt to avoid artifacts ---
    negative_prompt = (
        # --- Composition & Quality Issues ---
        "ugly, tiling, poorly drawn, out of frame, blurry, low contrast, bad art, amateur, "
        "frame border, signature, watermark, username, error, text, letters, writing, "
        
        # --- Anatomical & Deformity Issues ---
        "disfigured, deformed, body out of frame, bad anatomy, mutated, mutilated, "
        "extra limbs, extra arms, extra legs, extra fingers, too many fingers, fused fingers,"
        "asymmetric face, facial hair, crossed eyes, closed or missing eyes, fused eyes, distorted head, head detached, "
        "malformed limbs, bad hands, mutated hands, poorly drawn hands, "
        "floating head, head on chest, body parts on wrong places, incorrect anatomy, "

        # --- Duplication & Other Artifacts ---
        "duplicate, morbid, gross, unsettling, cloned face, draft"
    )

    # Anti-humanization if any non-human character
    if any(char.get("species") != "human" for char in updated_cast):
        negative_prompt += ", anthropomorphic, human-like features, standing on two legs, wearing clothes"

    scene_seed = seeds[0] if seeds else None    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "seed": scene_seed
        },
        "options": {"wait_for_model": True}
    }

    print(f"Generating image for prompt:\n{full_prompt}\nSeed used: {scene_seed}\n")

    # --- Step 4: Call the API with retry logic ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            image.save(filename)
            print(f"Image saved: {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Image generation failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Creating a placeholder image.")
                Image.new('RGB', (1024, 1024), color='grey').save(filename)
                return False

    return False