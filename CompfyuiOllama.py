import ollama
from ollama import Client
from PIL import Image
import numpy as np
import base64
from io import BytesIO

class OllamaVision:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "query": ("STRING", {
                    "multiline": True,
                    "default": "describe the image"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "llava"
                }),
                
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)

    FUNCTION = "vision"

    #OUTPUT_NODE = False

    CATEGORY = "Ollama"

    def vision(self, images, query, debug, url, model):
        images_b64 = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue())
            images_b64.append(str(img_bytes, 'utf-8'))
        if debug == "enable":
            print(f"""Your input contains:
                query: {query}
                url: {url}
                model: {model}
            """)
        #do some processing on the image, in this example I just invert it
        client = Client(host=url)
        response = client.chat(model=model, messages=[
          {
            'role': 'user',
            'content': query,
            'images': images_b64,
          },
        ])
        return (response['message']['content'],)

NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
}
