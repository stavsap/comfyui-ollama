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
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(self, images, query, debug, url, model):
        images_b64 = []

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue())
            images_b64.append(str(img_bytes, 'utf-8'))

        client = Client(host=url)

        response = client.generate(model=model, prompt=query, images=images_b64)

        if debug == "enable":
            print(f"""[Ollama Vision] query params:
                query: {query}
                url: {url}
                model: {model}
            """)

        return (response['response'],)


class OllamaGenerate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "llama2"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model):

        client = Client(host=url)

        response = client.generate(model=model, prompt=prompt)

        if debug == "enable":
            print(f"""[Ollama Generate] query params:
                prompt: {prompt}
                url: {url}
                model: {model}
            """)

        return (response['response'],)


NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaGenerate": OllamaGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaGenerate": "Ollama Generate",
}
