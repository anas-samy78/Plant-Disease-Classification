import io

import streamlit as st
from google import genai
from google.genai import types

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def build_prompt(species_name, detected_disease):
    return f"""
You are an expert botanist and plant pathologist with extensive experience in
horticulture and home gardening. You are assisting a non-expert gardener.

## Context from automated models (treat as hints, not ground truth)
- Species suggested by the identification model: {species_name}
- Condition suggested by the disease-detection model: {detected_disease}

## Instructions
1. Examine the provided image carefully. The image is your primary source of
   evidence; the model suggestions above are secondary.
2. If a suggestion is "Unknown", identify the plant or assess its condition
   yourself from the image.
3. If a suggestion clearly contradicts what you see in the image, trust the
   image and briefly note the discrepancy.
4. Disease labels may appear in a raw format (e.g. "Tomato___Late_blight").
   Interpret them as plain, readable names.
5. Do not overstate certainty. If the image quality, angle, or lighting limits
   your assessment, say so and mention what additional photo would help.
6. If the image does not show a plant, say so briefly and stop.

## Required response format
Use exactly the following sections, in this order, with these exact headings:

**Plant Overview**
Two to three sentences: the plant's common and scientific name (if identifiable),
its typical habitat, and one notable characteristic.

**Health Assessment**
State clearly whether the plant looks healthy, mildly affected, or severely
affected. Cite the specific visible evidence (leaf color, spots, texture,
wilting, etc.).

**Diagnosis**
If a problem is present, name it and explain what it is, what causes it, and how
it spreads, in simple language. If the plant is healthy, state that no disease
is evident.

**Treatment**
Numbered, actionable steps in order of priority (e.g. pruning, watering changes,
organic options first, then chemical options if necessary). If the plant is
healthy, write "No treatment needed."

**Prevention & Care**
Three to four concise tips covering watering, light, soil, and airflow as
relevant to this plant.

## Style
- Professional, clear, and reassuring; avoid unexplained jargon.
- Keep the whole response under 300 words.
- End with one short line advising consultation with a local agricultural
  expert for severe or uncertain cases.
"""


def get_plant_analysis(image, species_name, detected_disease):
    prompt = build_prompt(species_name, detected_disease)

    img = image.convert("RGB")
    img.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    image_bytes = buffer.getvalue()

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        prompt,
    ]

    last_error = None
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            if response.text:
                return response.text
        except Exception as e:
            last_error = e
            print(f"[Gemini] model {model_name} failed: {e}")  # بيظهر في logs

    raise last_error if last_error else RuntimeError("Empty response from Gemini")
