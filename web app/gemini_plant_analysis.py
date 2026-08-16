from google import genai
from PIL import Image
import streamlit as st
import base64

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def get_plant_analysis(image, species_name, detected_disease):

    prompt = f"""
You are a professional botanist and plant care expert.

Identified species: {species_name}
Detected condition: {detected_disease}

Analyze the provided plant image.

Give your response using exactly these sections:

**Plant Overview**
Brief description.

**Health Assessment**
Explain whether the plant appears healthy or shows signs of disease.

**Diagnosis**
Explain the detected disease in simple terms.

**Treatment**
Give practical treatment steps.

**Prevention & Care**
Give 3-4 concise prevention and care tips.

Be professional, concise, and easy for a non-expert gardener to understand.
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
