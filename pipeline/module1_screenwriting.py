import os
import re
import json
from typing import List, Dict, Any
import google.generativeai as genai

from pipeline.config import GOOGLE_API_KEY, ELEVEN_API_KEY, HF_API_TOKEN

# --- Environment Setup ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")


def _find_first_json_substring(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            stack.append('{')
        elif ch == '}':
            if not stack:
                continue
            stack.pop()
            if not stack:
                return text[start:i + 1]
    return None


def extract_json_from_response(response) -> Optional[dict]:
    """
    Robustly extract JSON dict from Gemini response (application/json or noisy text).
    """
    def try_parse_text(txt: str) -> Optional[dict]:
        if not txt:
            return None
        txt = txt.strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            pass
        js = _find_first_json_substring(txt)
        if js:
            try:
                return json.loads(js)
            except json.JSONDecodeError:
                # mild sanitization
                sanitized = js.replace("“", '"').replace("”", '"').replace("’", "'")
                sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
                try:
                    return json.loads(sanitized)
                except json.JSONDecodeError:
                    return None
        return None

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return try_parse_text(str(response))

    for cand in candidates:
        texts = []
        # new SDK shape
        try:
            content = getattr(cand, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            texts.append(t)
                else:
                    texts.append(str(content))
        except Exception:
            pass

        for attr in ("text", "output", "message", "message_text", "content"):
            try:
                if hasattr(cand, attr):
                    val = getattr(cand, attr)
                    texts.append(val if isinstance(val, str) else str(val))
            except Exception:
                pass

        texts.append(str(cand))

        for t in texts:
            parsed = try_parse_text(t)
            if parsed is not None:
                return parsed

    return None


def extract_text_from_response(response) -> Optional[str]:
    """
    Extract readable text from the Gemini response (for story plain-text).
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return str(response) if response is not None else None

    for cand in candidates:
        # new SDK shape
        try:
            content = getattr(cand, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts:
                    collected = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            collected.append(t)
                    if collected:
                        return " ".join(collected).strip()
                # else content may be string/dict-like
                txt = str(content)
                if txt:
                    return txt.strip()
        except Exception:
            pass
        # fallback attributes
        for attr in ("text", "output", "message", "message_text", "content"):
            try:
                if hasattr(cand, attr):
                    val = getattr(cand, attr)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            except Exception:
                pass
        # final stringify
        s = str(cand)
        if s:
            return s.strip()
    return None


def generate_story_text(rich_concept: str, cast_list: List[Dict[str, Any]]) -> str:
    source_context = cast_list[0].get("source_context") if cast_list else None
    FACTUAL_MODE = bool(source_context)
    mode_intro = (f"STRICT FACTUAL MODE for {source_context}: The story MUST adhere to widely accepted accounts/canon. "
                  "Do NOT invent events or characters. Visuals must be historically appropriate.") if FACTUAL_MODE else \
                 "CREATIVE MODE: You may imagine gentle magical elements while staying culturally respectful."

    prompt = (
        f"You are an Indian children's cultural storyteller. Through your stories, children learn positive life lessons.\n"
        f"{mode_intro}\n\n"
        f"Based on this concept: '{rich_concept}', write a story following the rules below - \n"
        f"--- RULES ---\n"
        f"- The story should be suitable for ages 6 to 10, simple & clear, with children friendly vocabulary.\n"
        f"- Make sure the story has a beginning, middle, and an explicit positive moral learning for a child at the end, derived from the story that a child can follow in life.\n"
        f"- Story Length: The total story must be strictly between 150 and 200 words to fit a 1.5 minute video.\n"
        f"- Total scenes: 6 to 8 (inclusive).\n\n"
        f"Return ONLY the story text as a single, clean paragraph, with no title or extra formatting."
    )

    print("[Module 1 - Step A] Requesting story paragraph...")
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "text/plain"})
    except Exception as e:
        print(f"[Module 1 - Step A] model.generate_content raised: {e}. Returning fallback short paragraph.")
        return f"{rich_concept}. A short fallback story."

    story_text = extract_text_from_response(response) or ""
    # sanitize into a single paragraph: collapse newlines and repeated spaces
    story_text = re.sub(r"\s+", " ", story_text).strip()
    # return as-is; caller may validate length
    print(f"[Module 1 - Step A] Story generated ({len(story_text.split())} words approx).")
    return story_text


def _rotate_shot_type(prev: Optional[str]) -> str:
    order = ["Wide Angle Shot", "Medium Shot", "Close up Shot"]
    if prev is None:
        return "Wide Angle Shot"
    for s in order:
        if s != prev:
            return s
    return "Medium Shot"


def generate_scene_list_from_story(story_text: str, cast_list: List[Dict[str, Any]], rich_concept: str) -> List[Dict[str, Any]]:
    """
    Convert the clean story paragraph into scene_list JSON following the scene schema.
    Returns: list of scene dicts (or fallback minimal scenes on parse failure).
    """
    character_names = ", ".join([char.get("name", "a character") for char in cast_list])
    source_context = cast_list[0].get("source_context") if cast_list else None
    FACTUAL_MODE = bool(source_context)
    mode_intro = (f"STRICT FACTUAL MODE for {source_context}: The story MUST adhere to widely accepted accounts/canon. "
                  "Do NOT invent events or characters. Visuals must be historically appropriate.") if FACTUAL_MODE else \
                 "CREATIVE MODE: You may imagine gentle magical elements while staying culturally respectful."

    # Example JSON template to force correct shape
    example_scene = {
        "scene_list": [
            {
                "scene_id": "scene_1",
                "sentence": "A short simple line describing the first beat.",
                "characters": [
                    {
                        "name": cast_list[0]["name"] if cast_list else "Character",
                        "identity_tag": cast_list[0].get("identity_tag", "char_canonical") if cast_list else "char_canonical",
                        "seed": cast_list[0].get("seed", None) if cast_list else None,
                        "visual_anchors": cast_list[0].get("visual_anchors", []) if cast_list else [],
                        "face_anchors": cast_list[0].get("face_anchors", []) if cast_list else [],
                        "age_stage": "child"
                    }
                ],
                "shot": ["wide establishing shot of the village"],
                "shot_type": "Wide Angle Shot"
            }
        ]
    }
    example_json = json.dumps(example_scene, ensure_ascii=False)

    prompt = (
        f"You are given a clean story paragraph and an enriched cast list. Break the story into a JSON object with key 'scene_list'.\n"
        f"{mode_intro}\n\n"
        f"Story:\n{story_text}\n\n"
        f"--- RULES ---\n"
        f"- Structure: Break the story into short and simple sentences.\n"
        f"- Preserve enriched characters (identity_tag, seed, visual_anchors, face_anchors). Do NOT overwrite them.\n"
        f"- Each active character MUST include an 'age_stage' field (child, youth, adult, elder) based on the active scene. Overwrite the existing age_stage as per the active shot.\n"
        f"--- SCENE OBJECT KEYS ---\n"
        f"1. 'scene_id': string like 'scene_1'.\n"
        f"2. 'sentence': a short narration describing the scene.\n"
        f"3. 'characters': subset of the cast active in this scene. Each character MUST have: name, identity_tag, seed, visual_anchors, face_anchors, age_stage.\n"
        f"4. 'shot': a single UNIQUE descriptive visual prompts describing character, action and the environment. Describe all three elements well.\n"
        f"5. 'shot_type': one of: Wide Angle Shot, Medium Shot, Close up Shot. Start with a Wide Angle shot for context and do NOT repeat the same shot_type consecutively.\n\n"
        f"Return ONLY the valid JSON object; NOTHING else. Example:\n{example_json}\n"
    )

    print("[Module 1 - Step B] Requesting scene_list JSON...")
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    except Exception as e:
        print(f"[Module 1 - Step B] model.generate_content raised: {e}. Falling back.")
        response = None

    # # debug dump
    # try:
    #     print("[Module 1 - Step B] RAW model response (truncated):")
    #     print(str(response)[:4000])
    # except Exception:
    #     pass

    try:
        data = None if response is None else extract_json_from_response(response)

        if not data or "scene_list" not in data:
            raise ValueError("Invalid or missing 'scene_list' in AI response")

        scene_list = data["scene_list"]

        # validate shape
        if not scene_list or not all(isinstance(s, dict) and "sentence" in s for s in scene_list):
            raise ValueError("AI response missing required keys in scene_list.")

        # Build maps for anchors, identity, seeds, ages
        anchors_map = {c["name"]: c.get("visual_anchors", []) for c in cast_list}
        face_map = {c["name"]: c.get("face_anchors", []) for c in cast_list}
        id_map = {c["name"]: c.get("identity_tag") for c in cast_list}
        seed_map = {c["name"]: c.get("seed") for c in cast_list}
        age_map = {c["name"]: c.get("age_stage", "unspecified") for c in cast_list}

        prev_shot_type = None
        for idx, s in enumerate(scene_list, start=1):
            s.setdefault("scene_id", f"scene_{idx}")

            # ensure shot is a list
            if "shot" in s and isinstance(s["shot"], str):
                s["shot"] = [s["shot"]]

            # fix or enforce shot_type non-repetition
            st = s.get("shot_type")
            if not st or st == prev_shot_type:
                # pick a shot_type different from prev
                chosen = _rotate_shot_type(prev_shot_type)
                s["shot_type"] = chosen
            prev_shot_type = s["shot_type"]

            # ensure characters have required keys and preserve anchors/seeds/identity
            new_chars = []
            for ch in s.get("characters", []):
                name = ch.get("name")
                new_ch = {
                    "name": name,
                    "identity_tag": id_map.get(name, ch.get("identity_tag", name.lower().replace(" ", "_") + "_canonical")),
                    "seed": seed_map.get(name, ch.get("seed")),
                    "visual_anchors": anchors_map.get(name, ch.get("visual_anchors", [])),
                    "face_anchors": face_map.get(name, ch.get("face_anchors", [])),
                    "age_stage": ch.get("age_stage", age_map.get(name, "unspecified"))
                }
                new_chars.append(new_ch)
            s["characters"] = new_chars

        print(f"[Module 1 - Step B] Scene list generated with {len(scene_list)} scenes.")
        return scene_list

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Module 1 - Step B] Failed to parse AI response: {e}. Falling back to minimal scenes.")
        # debug candidate snippets
        if response is not None:
            for cand in getattr(response, "candidates", [])[:5]:
                try:
                    print(">>> candidate:", getattr(cand, "content", str(cand))[:1000])
                except Exception:
                    print(">>> candidate (string):", str(cand)[:1000])

        # fallback valid scenes (2 scenes)
        fallback_scenes = []
        for idx in range(1, 3):
            fallback_scenes.append({
                "scene_id": f"scene_{idx}",
                "sentence": f"Scene {idx} about {rich_concept}",
                "characters": [
                    {
                        **c,
                        "identity_tag": c.get("identity_tag", c["name"].lower().replace(" ", "_") + "_canonical"),
                        "seed": c.get("seed"),
                        "visual_anchors": c.get("visual_anchors", []),
                        "face_anchors": c.get("face_anchors", []),
                        "age_stage": c.get("age_stage", "unspecified")
                    } for c in cast_list
                ],
                "shot": ["wide establishing shot"],
                "shot_type": "Wide Angle Shot"
            })
        return fallback_scenes
