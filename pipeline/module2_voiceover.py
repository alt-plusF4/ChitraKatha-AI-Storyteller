import os
import time
from elevenlabs.client import ElevenLabs
from translate import Translator
import numpy as np
import soundfile as sf
from pipeline.config import GOOGLE_API_KEY, ELEVEN_API_KEY, HF_API_TOKEN

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
client = ElevenLabs(api_key=ELEVEN_API_KEY)

def generate_audio(text: str, lang: str, filename: str, story_tone: str = None):
    """
    Generates narration audio in the user's chosen language.
    Returns True on success and False on failure.
    """
    # --- Step 1: Prepare narration text in the selected language ---
    print(f"Preparing narration in language: {lang}")
    narration_text = text # Default to English
    if lang.lower() != "english":
        print(f"Translating the Story to {lang}..")
        prompt = (
            f"Translate the following English sentence for a children's story into natural, fluent {lang}. "
            f"Use simple, easy-to-understand vocabulary. The original text is: '{text}'.\n"
            "IMPORTANT: Your entire response must be ONLY the translated text and nothing else. Do not add any explanations, options, or conversational filler."
        )
        response = gemini_model.generate_content(prompt)
        narration_text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                narration_text = "".join(part.text for part in candidate.content.parts if part.text)
        narration_text = response.text.strip()

    # --- Step 2: Generate audio with ElevenLabs API ---
    print("Generating high-quality narration via ElevenLabs...")
    

    voice_id = "MF4J4IDTRo0AxOO4dpFR" 

    # Adjust voice settings based on the desired story tone for expressiveness
    if story_tone == "Bedtime":
        voice_settings = {"stability": 0.95, "similarity_boost": 0.8, "style": 0.05, "speed": 0.85}
    elif story_tone == "Playtime":
        voice_settings = {"stability": 0.40, "similarity_boost": 0.75, "style": 0.30, "speed": 1.05}
    elif story_tone == "Educational":
        voice_settings = {"stability": 0.90, "similarity_boost": 0.75, "style": 0.15, "speed": 0.85}
    else:
        voice_settings = {"stability": 0.75, "similarity_boost": 0.8, "style": 0.1, "speed": 1.00}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Call the ElevenLabs API to convert the narration text to speech
            audio_stream = client.text_to_speech.convert(
                voice_id=voice_id,
                text=narration_text,
                model_id="eleven_multilingual_v2",
                voice_settings=voice_settings,
            )

            # Write the received audio stream to a file
            with open(filename, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)
            print(f"Audio saved: {filename}")
            return True # Indicate success

        except Exception as e:
            print(f"ElevenLabs API call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Creating silent audio as a fallback.")
                # --- Fallback: Create a silent audio file to prevent crashes ---
                # Estimate duration based on text length (average reading speed)
                estimated_duration_sec = len(text) / 15.0 
                sampling_rate = 24000
                silent_audio = np.zeros(int(estimated_duration_sec * sampling_rate), dtype=np.int16)
                sf.write(filename, silent_audio, sampling_rate)
                return True # Return True so the pipeline can continue with the silent clip
                
    return False # Indicate failure after all retries
