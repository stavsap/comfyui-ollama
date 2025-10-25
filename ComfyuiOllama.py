import random
import re

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


# ✅ Fixed model-list endpoint so dropdown populates correctly
@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    client = Client(host=url)

    try:
        response = client.list()
        models = response.get("models", [])
        model_names = []

        for m in models:
            # Prefer 'model' key if available; fall back to 'name'
            name = m.get("model") or m.get("name")
            if name:
                model_names.append(name)

        if not model_names:
            raise ValueError("No model names found in response.")

        return web.json_response(model_names)

    except Exception as e:
        return web.json_response({"error": str(e), "models": []})


class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
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
        image = Image.new('RGB', (100, 100), (255, 255, 255))
        image.save(path + ".png", pnginfo=metadata)
        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required": {"context_file": (files, {})}}

    CATEGORY = "Ollama"
    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1}),
                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1}),
                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1}),
                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1}),
                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05}),
                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05}),
                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False}),
                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"

    def ollama_options(self, **kargs):
        if kargs.get('debug'):
            print("--- ollama options v2 dump")
            pprint(kargs)
            print("---------------------------------------------------------")
        return (kargs,)


class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {
                    "tooltip": "Select a model for inference. This will list models available on the Ollama server."
                }),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }
        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?"
                }),
                "think": ("BOOLEAN", {"default": False}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "format": (["text", "json"], {}),
            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False}),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False}),
                "images": ("IMAGE", {"forceInput": False}),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False}),
                "meta": ("OLLAMA_META", {"forceInput": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"

    def get_request_options(self, options):
        response = None
        if options is None:
            return response
        enablers = ['enable_mirostat', 'enable_mirostat_eta',
                    'enable_mirostat_tau', 'enable_num_ctx',
                    'enable_repeat_last_n', 'enable_repeat_penalty',
                    'enable_temperature', 'enable_seed', 'enable_stop',
                    'enable_tfs_z', 'enable_num_predict',
                    'enable_top_k', 'enable_top_p', 'enable_min_p']
        for enabler in enablers:
            if options[enabler]:
                if response is None:
                    response = {}
                key = enabler.replace("enable_", "")
                response[key] = options[key]
        return response

    def ollama_generate_v2(self, system, prompt, think, keep_context, format,
                           context=None, options=None, connectivity=None, images=None, meta=None):
        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None:
                meta["connectivity"] = connectivity
            if options is not None:
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        url = meta['connectivity']['url']
        model = meta['connectivity']['model']
        client = Client(host=url)
        debug_print = bool(meta['options'] and meta['options'].get('debug'))

        if format == "text":
            format = ''

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit = 'm' if meta['connectivity']['keep_alive_unit'] == "minutes" else 'h'
        request_keep_alive = str(meta['connectivity']['keep_alive']) + keep_alive_unit
        request_options = self.get_request_options(options)

        images_b64 = None
        if images is not None:
            images_b64 = []
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                images_b64.append(str(img_bytes, 'utf-8'))

        response = client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images_b64,
            context=context,
            think=think,
            options=request_options,
            keep_alive=request_keep_alive,
            format=format,
        )

        ollama_response_text = response['response']
        ollama_response_thinking = response.get('thinking') if think else None

        if keep_context:
            self.saved_context = response["context"]

        return ollama_response_text, ollama_response_thinking, response['context'], meta,


NODE_CLASS_MAPPINGS = {
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaOptionsV2": "Ollama Options",
    "OllamaConnectivityV2": "Ollama Connectivity",
    "OllamaGenerateV2": "Ollama Generate",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
}
