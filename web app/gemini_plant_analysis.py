from google import genai
from PIL import Image
import streamlit as st
import base64

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def get_plant_analysis(image, species_name, detected_disease):

    prompt = f"""
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

    # Convert PIL image to bytes
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image",
                "data": image_b64,
                "mime_type": "image/jpeg"
            }
        ]
    )

    return interaction.output_text
