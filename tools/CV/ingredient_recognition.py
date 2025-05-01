import os
import openai
import base64

# 1. Set your API key in the environment variable OPENAI_API_KEY
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 2. Read and encode your image
def ingredients_detector(image_path: str) -> str:
    """
    Detects ingredients in the image using OpenAI's vision-enabled chat model.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: The assistant's response regarding the detected ingredients.
    """
    # Read the image and encode it in base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Build the messages payload
    messages = [
        {
            "role": "system",
            "content": (
                "You are a vision-enabled culinary assistant.\n"
                "• Detect every visible food ingredient in the image.\n"
                "• Return **only** a JSON array, no extra text. Each element must be a string.\n"
                "  Example output: [\"egg\", \"green bell pepper\", \"sesame oil\"]\n"
                "• If nothing is clearly recognizable, return an empty JSON array: []."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }
                }
            ]
        }
    ]


    # Call the vision-enabled chat model
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract and return the assistant’s answer
    return response.choices[0].message.content



print(ingredients_detector("/home/kree/Chopsticks-Dreams/data/pic1.jpg"))
