import random
import re

from ollama import Client
import numpy as np
from io import BytesIO
from pprint import pprint
from PIL import Image

# Collection of deprecated V1 nodes to be removed later.

class OllamaVision:
    DEPRECATED = True
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
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
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json", ''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(self, images, query, debug, url, model, seed, keep_alive, format):
        images_binary = []

        if format == "text":
            format = ''

        for (batch_number, image) in enumerate(images):
            # Convert tensor to numpy array
            i = 255. * image.cpu().numpy()

            # Create PIL Image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Save to BytesIO buffer
            buffered = BytesIO()
            img.save(buffered, format="PNG")

            # Get binary data
            img_binary = buffered.getvalue()
            images_binary.append(img_binary)

        client = Client(host=url)

        options = {
            "seed": seed
        }

        if debug == "enable":
            print(f"""[Ollama Vision]
request query params:

- query: {query}
- url: {url}
- model: {model}
- keep_alive: {keep_alive} minutes
- options: {options}
- format: {format}

""")
        
    

        response = client.generate(model=model, prompt=query, images=images_binary, options=options, keep_alive=str(keep_alive) + "m", format=format)

        if debug == "enable":
            print("[Ollama Vision]\nResponse:\n")
            pprint(response)

        return (response['response'],)


class OllamaGenerate:
    DEPRECATED = True
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
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json", ''],),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model, keep_alive, format, filter_thinking):

        client = Client(host=url)

        if format == "text":
            format = ''

        if debug == "enable":
            print(f"""[Ollama Generate]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}

            """)

        response = client.generate(model=model, prompt=prompt, keep_alive=str(keep_alive) + "m", format=format)

        if debug == "enable":
            print("[Ollama Generate]\nResponse:\n")
            pprint(response)
        
        ollama_response_text = response['response']
        if filter_thinking:
            ollama_response_text = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", ollama_response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return (ollama_response_text,)


# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

class OllamaGenerateAdvance:
    DEPRECATED = True
    saved_context = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                    "title": "system"
                }),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "format": (["text", "json", ''],),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            }, "optional": {
                "context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "context",)
    FUNCTION = "ollama_generate_advance"
    CATEGORY = "Ollama"

    def ollama_generate_advance(self, prompt, debug, url, model, system, seed, top_k, top_p, temperature, num_predict,
                                tfs_z, keep_alive, keep_context, format, filter_thinking, context=None):

        client = Client(host=url)

        if format == "text":
            format = ''

        # num_keep: int
        # seed: int
        # num_predict: int
        # top_k: int
        # top_p: float
        # tfs_z: float
        # typical_p: float
        # repeat_last_n: int
        # temperature: float
        # repeat_penalty: float
        # presence_penalty: float
        # frequency_penalty: float
        # mirostat: int
        # mirostat_tau: float
        # mirostat_eta: float
        # penalize_newline: bool
        # stop: Sequence[str]

        options = {
            "seed": seed,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "num_predict": num_predict,
            "tfs_z": tfs_z,
        }

        if context != None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context == None:
            context = self.saved_context

        if debug:
            print(f"""[Ollama Generate Advance]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- options: {options}
""")

        response = client.generate(model=model, system=system, prompt=prompt, context=context, options=options,
                                   keep_alive=str(keep_alive) + "m", format=format)
        if debug:
            print("[Ollama Generate Advance]\nResponse:\n")
            pprint(response)

        if keep_context:
            self.saved_context = response["context"]

        ollama_response_text = response['response']
        if filter_thinking:
            ollama_response_text = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", ollama_response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return (ollama_response_text, response['context'],)



NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision (deprecated)",
    "OllamaGenerate": "Ollama Generate (deprecated)",
    "OllamaGenerateAdvance": "Ollama Generate Advance (deprecated)",
}
