# Studio42 • OllamaSongwriterV3 (Songwriter Node Upgrade)
# Author: Willie & The Algorithms / Studio42
# License: MIT
#
# This version cleans model output into proper lyric formatting.
# Outputs:
#   1. style_prompt_output – production/arrangement style
#   2. lyrics_output – structured, properly formatted lyrics
#   3. raw_response – unmodified LLM response for review/logging

from __future__ import annotations
import json
import re
from typing import Any, Dict, Tuple, Optional
from ollama import Client

# -------------------------------------------------------------------
# Default ACE Songwriter System Prompt
# -------------------------------------------------------------------
DEFAULT_SYSTEM = """You are a professional songwriter and lyricist who has written for Green Day, Blink-182, and Dishwalla.
You are now collaborating with a new artist called "Willie & The Algorithms" — an experimental alt-punk act blending raw guitar energy, melodic melancholy, and themes of digital identity, nostalgia, and human connection.

Follow ACE-step formatting internally to guide your process, but OUTPUT MUST BE STRICT JSON (no commentary).
- A: Write stage-ready, emotionally honest, rhythmically tight songs.
- C: Style fusion: Green Day (anthemic bite), Blink-182 (youthful urgency, hooks), Dishwalla (introspective warmth), Willie & The Algorithms (analog hearts / digital ghosts).
- E: Rules
  1) Structure lyrics with labeled sections: [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro].
  2) Each label on its own line, no literal "\\n" characters.
  3) Use natural line breaks and consistent syllabic rhythm.
  4) End with a short production note paragraph (<120 words).
  5) Output STRICT JSON ONLY:
{
  "style": "Production and stylistic guidance (tempo/BPM, vibe, instrumentation).",
  "lyrics": "[Verse 1]\\n...\\n[Chorus]\\n...",
  "full_response": "Repeat the full text you produced (style + lyrics)."
}"""

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _enabled_options(opts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not opts:
        return None
    enablers = [k for k in opts.keys() if k.startswith("enable_")]
    out = {}
    for e in enablers:
        if opts[e]:
            key = e.replace("enable_", "")
            out[key] = opts.get(key)
    return out or None

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try fenced ```json blocks
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # Try first { } block
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    return None

def _clean_lyrics(text: str) -> str:
    if not text:
        return ""
    # Convert escaped \n to real newlines
    text = text.replace("\\n", "\n")
    # Ensure section headers are isolated
    text = re.sub(r"(\[.*?\])", r"\n\1\n", text)
    # Remove duplicate blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _split_fallback(raw: str) -> Tuple[str, str]:
    if not raw:
        return "", ""
    # Detect section headers to split style/lyrics
    match = re.search(r"^\s*\[.*?\]", raw, re.MULTILINE)
    if match:
        idx = match.start()
        return raw[:idx].strip(), raw[idx:].strip()
    # Default: first paragraph = style
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(parts) > 1:
        return parts[0], "\n\n".join(parts[1:])
    return "", raw.strip()

# -------------------------------------------------------------------
# Node class
# -------------------------------------------------------------------
class OllamaSongwriterV3:
    saved_context = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM}),
                "style_prompt": ("STRING", {"multiline": True, "default": "Alt-punk with trip-hop undertones. Neon/lo-fi vibe. Themes: connection vs. isolation in a digital city at night."}),
                "lyrics_prompt": ("STRING", {"multiline": True, "default": "Song about losing your reflection online but finding your voice."}),
            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False}),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("style_prompt_output", "lyrics_output", "raw_response")
    FUNCTION = "generate"
    CATEGORY = "Ollama/Songwriting"
    DESCRIPTION = "Generates style guidance + formatted lyrics + full raw LLM response."

    def generate(
        self,
        system_prompt: str,
        style_prompt: str,
        lyrics_prompt: str,
        connectivity=None,
        options=None,
        keep_context=False,
        context=None,
    ):
        if not connectivity:
            raise Exception("OllamaSongwriterV3 requires a 'connectivity' input from OllamaConnectivityV2.")
        url = connectivity["url"]
        model = connectivity["model"]
        keep_alive_unit = "m" if connectivity.get("keep_alive_unit") == "minutes" else "h"
        keep_alive_val = f"{connectivity.get('keep_alive', 5)}{keep_alive_unit}"

        client = Client(host=url)
        if isinstance(context, str):
            try:
                context = [int(x.strip()) for x in context.split(",") if x.strip()]
            except Exception:
                context = None
        if keep_context and context is None:
            context = self.saved_context

        req_opts = _enabled_options(options)

        user_prompt = f"""STYLE INPUT:
{style_prompt}

LYRICS INPUT:
{lyrics_prompt}

Output STRICT JSON matching the given schema. Do not add commentary or quotes around sections.
"""

        response = client.generate(
            model=model,
            system=system_prompt,
            prompt=user_prompt,
            context=context,
            options=req_opts,
            keep_alive=keep_alive_val,
            format="",
        )

        raw = response.get("response", "") or ""
        if keep_context:
            self.saved_context = response.get("context")

        data = _extract_json(raw)
        if data and isinstance(data, dict):
            style_out = _clean_lyrics(data.get("style", ""))
            lyrics_out = _clean_lyrics(data.get("lyrics", ""))
            full_out = data.get("full_response", raw)
            return (style_out, lyrics_out, full_out.strip())

        # fallback if not JSON
        style_out, lyrics_out = _split_fallback(raw)
        return (_clean_lyrics(style_out), _clean_lyrics(lyrics_out), raw.strip())


NODE_CLASS_MAPPINGS = {
    "OllamaSongwriterV3": OllamaSongwriterV3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaSongwriterV3": "Ollama • Songwriter V3 (Style + Lyrics + Raw)",
}
