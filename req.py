import google.generativeai as genai
from PIL import Image
from io import BytesIO

# Configure API key
genai.configure(api_key="Gemini-API-key")  # Replace with your actual API key

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")

# Prompt for image generation
contents = input("Enter your prompt for image generation: ")

# Generate content using Gemini model
response = model.generate_content(
    contents,
    generation_config={"response_modalities": ['TEXT', 'IMAGE']}
)

# Process response
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)  # Display any accompanying text
    if part.inline_data is not None and part.inline_data.data:
        try:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save('gemini-generated-image.png')  # Save the generated image
            image.show()  # Show the generated image
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print("No image data received.")
