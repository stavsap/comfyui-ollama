import random

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os

@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    client = Client(host=url)
    models = [model['model'] for model in client.list().get('models', [])]
    
    return web.json_response(models)

class OllamaVision:
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
                "format": (["text", "json",''],),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(self, images, query, debug, url, model, seed, keep_alive, format):
        images_b64 = []

        if format == "text":
            format = ''

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue())
            images_b64.append(str(img_bytes, 'utf-8'))

        client = Client(host=url)

        if debug == "enable":
            print(f"""[Ollama Vision]
request query params:

- query: {query}
- url: {url}
- model: {model}

""")

        response = client.generate(model=model, prompt=query, images=images_b64, keep_alive=str(keep_alive) + "m", format=format)

        if debug == "enable":
            print("[Ollama Vision]\nResponse:\n")
            pprint(response)

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
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "format": (["text", "json",''],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model, keep_alive, format):

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

        return (response['response'],)

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

class OllamaGenerateAdvance:
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
                    "title":"system"
                }),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "format": (["text", "json",''],),
            },"optional": {
                "context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "context",)
    FUNCTION = "ollama_generate_advance"
    CATEGORY = "Ollama"

    def ollama_generate_advance(self, prompt, debug, url, model, system, seed, top_k, top_p,temperature, num_predict, tfs_z, keep_alive, keep_context, format, context=None):

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
            "top_k":top_k,
            "top_p":top_p,
            "temperature":temperature,
            "num_predict":num_predict,
            "tfs_z":tfs_z,
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

        response = client.generate(model=model, system=system, prompt=prompt, context=context, options=options, keep_alive=str(keep_alive) + "m", format=format)
        if debug:
            print("[Ollama Generate Advance]\nResponse:\n")
            pprint(response)

        if keep_context:
            self.saved_context = response["context"]

        return (response['response'], response['context'],)

class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True}, ),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path+".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)

NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
}
