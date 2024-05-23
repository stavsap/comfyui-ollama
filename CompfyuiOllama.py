import random

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
                "keep_alive": ("INT", {"default": 5, "min": 0, "max": 60, "step": 5}),

            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "ollama_vision"
    CATEGORY = "Ollama"

    def ollama_vision(self, images, query, debug, url, model, keep_alive):
        images_b64 = []

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

        response = client.generate(model=model, prompt=query, images=images_b64, keep_alive=str(keep_alive) + "m")



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
                "keep_alive": ("INT", {"default": 5, "min": 0, "max": 60, "step": 5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model, keep_alive):

        client = Client(host=url)

        if debug == "enable":
            print(f"""[Ollama Generate]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}

            """)

        response = client.generate(model=model, prompt=prompt, keep_alive=str(keep_alive) + "m")

        if debug == "enable":
                print(f"""\n[Ollama Generate]
response:

- model: {response["model"]}
- created_at: {response["created_at"]}
- done: {response["done"]}
- eval_duration: {response["eval_duration"]}
- load_duration: {response["load_duration"]}
- eval_count: {response["eval_count"]}
- eval_duration: {response["eval_duration"]}
- prompt_eval_duration: {response["prompt_eval_duration"]}

- response: {response["response"]}

- context: {response["context"]}

""")

        return (response['response'],)

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

class OllamaGenerateAdvance:
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
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "llama2"
                }),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                    "title":"system"
                }),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_k": ("FLOAT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("FLOAT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": 0, "max": 60, "step": 5}),
            },"optional": {
                "context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("response","context",)
    FUNCTION = "ollama_generate_advance"
    CATEGORY = "Ollama"

    def ollama_generate_advance(self, prompt, debug, url, model, system, seed,top_k, top_p,temperature, num_predict, tfs_z, keep_alive, context=None):

        client = Client(host=url)

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

        if debug == "enable":
            print(f"""[Ollama Generate Advance]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- options: {options}
""")

        response = client.generate(model=model, system=system, prompt=prompt, context=context, options=options, keep_alive=str(keep_alive) + "m")

        if debug == "enable":
            print(f"""\n[Ollama Generate Advance]
response:

- model: {response["model"]}
- created_at: {response["created_at"]}
- done: {response["done"]}
- eval_duration: {response["eval_duration"]}
- load_duration: {response["load_duration"]}
- eval_count: {response["eval_count"]}
- eval_duration: {response["eval_duration"]}
- prompt_eval_duration: {response["prompt_eval_duration"]}

- response: {response["response"]}

- context: {response["context"]}

""")
        return (response['response'],response['context'],)

NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
}
