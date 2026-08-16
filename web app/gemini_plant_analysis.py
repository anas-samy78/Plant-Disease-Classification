import google.generativeai as genai
from PIL import Image

genai.configure(api_key=st.secrets["AQ.Ab8RN6ICjA79ZmlbLMbGNIaIk_kIhVcXFVkQu5DKoc2SKpMpBw"])
gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")


def get_plant_analysis(image, species_name, detected_disease):

    prompt = f"""
You are a professional botanist and plant care expert. Analyze the provided
plant image and give a structured, professional response.

Context from our detection models:
- Identified species: {species_name}
- Detected condition: {detected_disease}

Structure your response EXACTLY in this format, using clear section headers:

**Plant Overview**
A brief 2-3 sentence description of this species.

**Health Assessment**
State clearly whether the plant appears healthy or shows signs of disease,
based on the visual evidence in the image.

**Diagnosis**
If diseased, explain the disease in simple terms. If healthy, state that
no significant issues were detected.

**Treatment**
Practical, actionable treatment steps if the plant is diseased. If healthy,
write "No treatment needed."

**Prevention & Care**
3-4 concise tips for protecting this plant type from future issues and
general care recommendations.

Keep the tone professional but easy to understand for a non-expert gardener.
Avoid unnecessary repetition. Be concise but complete.
"""

    response = gemini_model.generate_content([prompt, image])
    return response.text

if __name__ == "__main__":
    img = Image.open("plant.jpg")
    result = get_plant_analysis(
        image=img,
        species_name="Malus domestica",
        detected_disease="Apple Scab"
    )
    print(result)
