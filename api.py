from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key="Gemini-API-Key")
model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")

class RecommendRequest(BaseModel):
    occasion: str
    weather: str

class PromptRequest(BaseModel):
    prompt: str

async def generate_image_for_item(prompt: str):
    response = model.generate_content(
        prompt,
        generation_config={"response_modalities": ['TEXT', 'IMAGE']}
    )
    result_image_b64 = None
    for part in response.candidates[0].content.parts:
        if getattr(part, 'inline_data', None) is not None and part.inline_data.data:
            image = Image.open(BytesIO(part.inline_data.data))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            result_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return result_image_b64

@app.post("/recommend")
async def recommend_outfit(data: RecommendRequest):
    items = [
        {"id": 1, "name": "Shirt", "type": "Top"},
        {"id": 2, "name": "Jeans", "type": "Bottom"},
        {"id": 3, "name": "Sneakers", "type": "Shoes"}
    ]
    # Generate images concurrently for all items
    prompts = [f"{item['name']} {item['type']} for {data.occasion} in {data.weather} weather, studio photo, white background" for item in items]
    images = await asyncio.gather(*[generate_image_for_item(prompt) for prompt in prompts])
    for item, img in zip(items, images):
        item["image"] = f"data:image/png;base64,{img}" if img else ""
    outfits = [
        {
            "id": 1,
            "title": f"Outfit for {data.occasion} ({data.weather})",
            "items": items,
            "occasion": data.occasion,
            "matchScore": 90,
            "description": f"A recommended outfit for {data.occasion} in {data.weather} weather."
        }
    ]
    return {"outfits": outfits}

@app.post("/generate")
async def generate_content(data: PromptRequest):
    response = model.generate_content(
        data.prompt,
        generation_config={"response_modalities": ['TEXT', 'IMAGE']}
    )
    result_text = ""
    result_image_b64 = None
    for part in response.candidates[0].content.parts:
        if getattr(part, 'text', None) is not None:
            result_text += part.text
        if getattr(part, 'inline_data', None) is not None and part.inline_data.data:
            image = Image.open(BytesIO(part.inline_data.data))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            result_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"text": result_text, "image": result_image_b64}
