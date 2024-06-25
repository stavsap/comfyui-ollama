import random

from ollama import Client
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web

import logging

RETRY_LIMIT=3

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ComfyUI-Ollama')


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    client = Client(host=url)
    models = [model['name'] for model in client.list().get('models', [])]
    
    return web.json_response(models)

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
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 360, "step": 1}),                
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

        if debug:
            logger.info(f"""[Ollama Vision]
request query params:

- query: {query}
- url: {url}
- model: {model}
""")
        else:
            logger.info("[Ollama Vision]: request query")
        
        for i in range(RETRY_LIMIT):
            try:
                response = client.generate(model=model, prompt=query, images=images_b64, keep_alive=str(keep_alive) + "m")
                _key_list=list(response.keys())
                if debug:
                    logger.info(f"""\n[Ollama Vision]
response:

- model: {response["model"] if "model" in _key_list else None}
- created_at: {response["created_at"]if "created_at" in _key_list else None}
- done: {response["done"]if "done" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- load_duration: {response["load_duration"]if "load_duration" in _key_list else None}
- eval_count: {response["eval_count"]if "eval_count" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- prompt_eval_duration: {response["prompt_eval_duration"]if "prompt_eval_duration" in _key_list else None}

- response: {response["response"]if "response" in _key_list else None}

- context: {response["context"]if "context" in _key_list else None}
""")
                else:
                    logger.info("[Ollama Vision]: get response")
                    if response['response']== '':raise KeyError('Empty Response')
                break #No Exception, break
            except Exception as e:
                logger.warn(f"""[Ollama Vision]
get response ERROR:
{e.__class__,e}""")
                if i+1 ==RETRY_LIMIT:
                    logger.error("[Ollama Vision]: retry over limit")
                    raise e
                else:
                    logger.info(f"[Ollama Vision]: retry {i+1}") #Retry, no break
                
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
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 360, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ollama_generate"
    CATEGORY = "Ollama"

    def ollama_generate(self, prompt, debug, url, model, keep_alive):

        client = Client(host=url)

        if debug:
            logger.info(f"""[Ollama Generate]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
""")
        else:
            logger.info("[Ollama Vision]: request query")
        for i in range(RETRY_LIMIT):
            try:
                response = client.generate(model=model, prompt=prompt, keep_alive=str(keep_alive) + "m")
                _key_list=list(response.keys())
                
                if debug:
                    logger.info(f"""[Ollama Generate]
response:

- model: {response["model"] if "model" in _key_list else None}
- created_at: {response["created_at"]if "created_at" in _key_list else None}
- done: {response["done"]if "done" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- load_duration: {response["load_duration"]if "load_duration" in _key_list else None}
- eval_count: {response["eval_count"]if "eval_count" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- prompt_eval_duration: {response["prompt_eval_duration"]if "prompt_eval_duration" in _key_list else None}

- response: {response["response"]if "response" in _key_list else None}

- context: {response["context"]if "context" in _key_list else None}
""")
                else:
                    logger.info("[Ollama Generate]: get response")
                    if response['response']== '':raise KeyError('Empty Response')
                break #No Exception, break
            except Exception as e:
                logger.warn(f"""[Ollama Generate]
get response ERROR:
{e.__class__,e}""")
                if i+1 ==RETRY_LIMIT:
                    logger.error("[Ollama Generate]: retry over limit")
                    raise e
                else:
                    logger.info(f"[Ollama Generate]: retry {i+1}") #Retry, no break
                
        return (response['response'],)

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
                "top_k": ("FLOAT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("FLOAT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 360, "step": 1}),
                "keep_context": ("BOOLEAN", {"default": False}),
            },"optional": {
                "context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("response","context",)
    FUNCTION = "ollama_generate_advance"
    CATEGORY = "Ollama"

    def ollama_generate_advance(self, prompt, debug, url, model, system, seed, top_k, top_p,temperature, num_predict, tfs_z, keep_alive, keep_context, context=None):

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

        if keep_context and context == None:
            context = self.saved_context

        if debug:
            logger.info(f"""[Ollama Generate Advance]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- options: {options}
""")
        else:
            logger.info("[Ollama Generate Advance]: request query")
        for i in range(RETRY_LIMIT):
            try:
                response = client.generate(model=model, system=system, prompt=prompt, context=context, options=options, keep_alive=str(keep_alive) + "m")
                _key_list=list(response.keys())

                if debug:
                    logger.info(f"""[Ollama Generate Advance]
response:

- model: {response["model"] if "model" in _key_list else None}
- created_at: {response["created_at"]if "created_at" in _key_list else None}
- done: {response["done"]if "done" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- load_duration: {response["load_duration"]if "load_duration" in _key_list else None}
- eval_count: {response["eval_count"]if "eval_count" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- prompt_eval_duration: {response["prompt_eval_duration"]if "prompt_eval_duration" in _key_list else None}

- response: {response["response"]if "response" in _key_list else None}

- context: {response["context"]if "context" in _key_list else None}
""")
                else:
                    logger.info("[Ollama Generate Advance]: get response")
                    if response['response']== '':raise KeyError('Empty Response')
                break #No Exception, break
            except Exception as e:
                logger.warn(f"""[Ollama Generate Advance]
get response ERROR:
{e.__class__,e}""")
                if i+1 ==RETRY_LIMIT:
                    logger.error("[Ollama Generate Advance]: retry over limit")
                    raise e
                else:
                    logger.info(f"[Ollama Generate Advance]: retry {i+1}") #Retry, no break
                    
        if keep_context:
            self.saved_context = response["context"]
            
        return (response['response'],response['context'],)
    
class OllamaOneRoundChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Tell me about Art!"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 360, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_once",)
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"

    def ollama_chat(self, system_prompt, user_prompt, debug, url, model, keep_alive):

        client = Client(host=url)
        messages=[
                    {
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content": user_prompt
                    }
                ]
        if debug:
            logger.info(f"""[Ollama One Round Chat]
request query params:

- message: {messages}
- url: {url}
- model: {model}
""")
        else:
            logger.info("[Ollama One Round Chat]: request query")
            
        for i in range(RETRY_LIMIT):
            try:
                response = client.chat(model=model, stream=False, messages=messages, keep_alive=str(keep_alive) + "m")
                _key_list=list(response.keys())

                if debug:
                    logger.info(f"""[Ollama One Round Chat]
response:

- model: {response["model"] if "model" in _key_list else None}
- created_at: {response["created_at"]if "created_at" in _key_list else None}
- done: {response["done"]if "done" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- load_duration: {response["load_duration"]if "load_duration" in _key_list else None}
- eval_count: {response["eval_count"]if "eval_count" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- prompt_eval_duration: {response["prompt_eval_duration"]if "prompt_eval_duration" in _key_list else None}

- response: {response['message']['content']if "message" in _key_list else None}
""")
                else:
                    logger.info("[Ollama One Round Chat]: get response")
                    if response['message']['content']== '':raise KeyError('Empty Response')#Check if ERROR here and raise
                break #No Exception, break
            except Exception as e:
                logger.warn(f"""[Ollama One Round Chat]
get response ERROR:
{e.__class__,e}""")
                if i+1 ==RETRY_LIMIT:
                    logger.error("[Ollama One Round Chat]: retry over limit")
                    raise e
                else:
                    logger.info(f"[Ollama One Round Chat]: retry {i+1}") #Retry, no break

        return (response['message']['content'],)

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion

class OllamaOneRoundChatAdvance:
    saved_context = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Tell me about Art!"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "model": ((), {}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "top_k": ("FLOAT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "num_predict": ("FLOAT", {"default": -1, "min": -2, "max": 2048, "step": 1}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 360, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_once",)
    FUNCTION = "ollama_chat_advance"
    CATEGORY = "Ollama"

    def ollama_chat_advance(self, system_prompt, user_prompt, debug, url, model, seed, top_k, top_p,temperature, num_predict, tfs_z, keep_alive):

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
        messages=[
                    {
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content": user_prompt
                    }
                ]
        if debug:
            logger.info(f"""[Ollama One Round Chat Advance]
request query params:

- message: {messages}
- url: {url}
- model: {model}
- options: {options}
""")
        else:
            logger.info("[Ollama One Round Chat Advance]: request query")
            
        for i in range(RETRY_LIMIT):
            try:
                response = client.chat(model=model, stream=False, messages=messages, options=options, keep_alive=str(keep_alive) + "m")
                _key_list=list(response.keys())
                
                if debug:
                    logger.info(f"""[Ollama One Round Chat Advance]
response:

- model: {response["model"] if "model" in _key_list else None}
- created_at: {response["created_at"]if "created_at" in _key_list else None}
- done: {response["done"]if "done" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- load_duration: {response["load_duration"]if "load_duration" in _key_list else None}
- eval_count: {response["eval_count"]if "eval_count" in _key_list else None}
- eval_duration: {response["eval_duration"]if "eval_duration" in _key_list else None}
- prompt_eval_duration: {response["prompt_eval_duration"]if "prompt_eval_duration" in _key_list else None}

- response: {response['message']['content']if "message" in _key_list else None}
""")
                else:
                    logger.info("[Ollama One Round Chat Advance]: get response")
                    if response['message']['content']== '':raise KeyError('Empty Response')#Check if ERROR here and raise
                break #No Exception, break
            except Exception as e:
                logger.warn(f"""[Ollama One Round Chat Advance]
get response ERROR:
{e.__class__,e}""")
                if i+1 ==RETRY_LIMIT:
                    logger.error("[Ollama One Round Chat Advance]: retry over limit")
                    raise e
                else:
                    logger.info(f"[Ollama One Round Chat Advance]: retry {i+1}") #Retry, no break

        return (response['message']['content'],)

NODE_CLASS_MAPPINGS = {
    "OllamaVision": OllamaVision,
    "OllamaGenerate": OllamaGenerate,
    "OllamaGenerateAdvance": OllamaGenerateAdvance,
    "OllamaOneRoundChat": OllamaOneRoundChat,
    "OllamaOneRoundChatAdvance": OllamaOneRoundChatAdvance,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVision": "Ollama Vision",
    "OllamaGenerate": "Ollama Generate",
    "OllamaGenerateAdvance": "Ollama Generate Advance",
    "OllamaOneRoundChat": "Ollama One Round Chat",
    "OllamaOneRoundChatAdvance": "Ollama One Round Chat Advance",

}
