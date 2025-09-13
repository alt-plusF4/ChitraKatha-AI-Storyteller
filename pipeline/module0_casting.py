import re
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from pipeline.config import GOOGLE_API_KEY, ELEVEN_API_KEY, HF_API_TOKEN



genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')


CONTEXT_KEYWORDS: Dict[str, List[str]] = {
    "Ramayana": [
        "rama", "sita", "hanuman", "ravana", "lanka", "ayodhya", "shri ram",
        "lakshmana", "bharata", "shatrughna", "valmiki", "sugriva", "vibhishana",
        "kumbhakarna", "maricha", "mandodari", "jatayu", "agastya", "ashoka vatika",
        "pushpaka viman", "vanarasena", "dandaka forest"
    ],
    "Mahabharata": [
        "pandava", "kaurava", "krishna", "arjuna", "draupadi", "kurukshetra", "shree krishna",
        "bhishma", "drona", "karna", "yudhishthira", "bheema", "nakula", "sahadeva",
        "shakuni", "duryodhana", "gandhari", "kunti", "vidura", "abhimanyu", "vyasa",
        "bhagavad gita", "eighteen days war"
    ],
    "Freedom Struggle": [
        "freedom fighter", "gandhi", "nehru", "bose", "1947",
        "bhagat singh", "chandrashekhar azaad", "lal bahadur shastri", "sardar patel",
        "subhash chandra bose", "quit india", "non-cooperation", "civil disobedience",
        "jallianwala bagh", "dandi march", "swadeshi", "revolutionary", "indian national congress",
        "partition", "mountbatten", "simon commission", "round table conference"
    ],
    "Indus Valley Civilization": [
        "mohenjo-daro", "harappa", "dholavira", "great bath", "citadel", "granary",
        "seal", "script", "urban planning", "drainage", "terracotta", "cotton", "beads", "sindhu"
    ],
    "Buddhism and Jainism": [
        "buddha", "siddhartha", "nirvana", "dhamma", "sangha", "ashoka",
        "tripitaka", "stupa", "viharas", "mahavira", "tirthankara", "ahimsa", "jina", "eightfold path"
    ],
}

CREATURE_KEYWORDS: List[str] = [
    "bee", "ant", "butterfly", "sparrow", "lion", "tiger", "elephant", "monkey",
    "dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken",
    "eagle", "falcon", "owl", "hawk", "dove", "peacock", "pigeon", "parrot",
    "shark", "dolphin", "whale", "seal", "otter", "penguin", "jellyfish", "octopus",
    "crab", "lobster", "shrimp", "starfish", "seahorse", "clam", "snail", "urchin",
    "frog", "toad", "newt", "salamander", "crocodile", "alligator", "turtle", "lizard",
    "snake", "gecko", "chameleon", "iguana", "kangaroo", "koala", "panda", "sloth",
    "wolf", "fox", "bear", "deer", "moose", "bison", "zebra", "giraffe",
    "rhino", "hippo", "camel", "donkey", "bat", "rat", "mouse", "squirrel",
    "raccoon", "skunk", "beaver", "porcupine", "hedgehog", "armadillo", "lemur", "meerkat",
    "spider", "scorpion", "centipede", "millipede", "firefly", "ladybug", "grasshopper", "cricket"
]


def _detect_context(text: str, keyword_map: Dict[str, List[str]]) -> Optional[str]:
    lowered_text = text.lower()
    best, best_hits = None, 0
    for context, keywords in keyword_map.items():
        hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered_text))
        if hits > best_hits:
            best, best_hits = context, hits
    return best

def enrich_prompt(simple_prompt: str, story_tone: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates a rich concept and a stable cast_list with immutable visual_anchors.
    """
    context = _detect_context(simple_prompt, CONTEXT_KEYWORDS)

    tone_map = {
        "Bedtime": "a soothing, gentle, and calming tone",
        "Playtime": "an adventurous, energetic, and exciting tone",
        "Educational": "an informative, clear, and engaging tone.",
        "Default": "a general storytelling tone, expressive and clear."
    }
    tone_instruction = tone_map.get(story_tone, "a heartwarming and age-appropriate tone")

    if context:
        mode_instruction = (
            f"STRICT FACTUAL MODE for {context}: "
            "The concept must align with widely accepted accounts/canon for this context. "
            "Do NOT invent events, places, or characters; avoid anachronisms or crossovers. STICK TO KNOWN FACTS."
        )
    else:
        mode_instruction = (
            "CREATIVE MODE: You may imagine new characters, places, and gentle magical elements as long as they are culturally respectful."
        )

    # --- tightened JSON schema: require species + visual_anchors (immutable) ---
    prompt = (
        f"You are a cultural historian, casting director, and cinematographer for children's stories. "
        f"Based on the idea: '{simple_prompt}', generate JSON with two keys: 'concept' and 'cast_list'.\n"
        f"Mode: {mode_instruction}\n"
        f"1. 'concept': A single, rich but concise paragraph (around 4-5 sentences) describing the story, suitable for a young child with {tone_instruction}.\n"
        f"2. 'cast_list': An array of character sheets. EACH character object MUST include exactly:\n"
        f"   - 'name' (string)\n"
        f"   - 'species' (string: 'human' or creature)\n"
        f"   - 'identity_tag' (string: immutable short unique identifier')\n"
        f"   - 'seed' (integer: a fixed random seed, ensures character consistency across scenes)\n"
        f"   - 'visual_anchors' (array of 3-5 immutable traits: gender, hairstyle, facial markers, body structure, skin colour, **symbolic items**)\n"
        f"   - 'face_anchors' (array of immutable facial traits: jawline, eyes, lips, nose) - Strictly **no facial hair** in immutable facial anchors for every character.\n"
        f"These anchors represent permanent identity traits that must remain unchanged. "
        f"Do NOT include mood or temporary states in anchors.\n"
        f"Return ONLY raw JSON (no markdown)."
    )

    print("[Module 0] Requesting Gemini to enrich prompt and create cast list...")
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        raw_parts = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            raw_parts.append(part.text)        
        cleaned = (response.text or "").strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned)
        rich_concept = data.get("concept", f"A beautiful story about {simple_prompt}")
        cast_list = data.get("cast_list", [])

        if not isinstance(cast_list, list) or not all(isinstance(c, dict) for c in cast_list):
            raise ValueError("'cast_list' is not valid.")

        # Enforce integrity + assign seeds if missing
        for c in cast_list:
            c["source_context"] = context
            c.setdefault("species", "human")
            c.setdefault("visual_anchors", [])
            c.setdefault("face_anchors", ["clear eyes", "defined lips"])
            c.setdefault("identity_tag", c.get("name", "").lower().replace(" ", "_") + "_canonical")
            c.setdefault("seed", random.randint(10000, 99999))

        print("Prompt and cast list enriched!")
        return rich_concept, cast_list

    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"[Module 0] Fallback due to error: {e}")
        fallback_concept = f"A story about {simple_prompt}."
        fallback_cast = [{
            "name": "A character",
            "species": "human",
            "identity_tag": "character_canonical",
            "visual_anchors": ["warm eyes", "kind smile", "simple clothing"],
            "face_anchors": ["clear lips", "round eyes"],
            "seed": random.randint(10000, 99999),
            "source_context": context
        }]
        return fallback_concept, fallback_cast