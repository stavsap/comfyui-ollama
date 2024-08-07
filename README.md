# ComfyUI Ollama

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) Nodes for interacting with [Ollama](https://ollama.com/) using the [ollama python client](https://github.com/ollama/ollama-python).

Integrate the power of LLMs into ComfyUI workflows easily or just experiment with GPT.

To use this properly, you would need a running Ollama server reachable from the host that is running ComfyUI.

## Installation

Use the [compfyui manager](https://github.com/ltdrdata/ComfyUI-Manager) "Custom Node Manager":

![pic](.meta/InstallViaManager.png)

Search `ollama` and select the one by `stavsap`

![pic](.meta/manager-install.png)

**Or**

1. git clone into the ```custom_nodes``` folder inside your ComfyUI installation or download as zip and unzip the contents to ```custom_nodes/compfyui-ollama```.
2. `pip install -r requirements.txt`
3. Start/restart ComfyUI

### Nodes

### OllamaVision

A node that gives an ability to query input images. 

![pic](.meta/OllamaVision.png)

A model name should be model with Vision abilities, for example: https://ollama.com/library/llava.

### OllamaGenerate

A node that gives an ability to query an LLM via given prompt. 

![pic](.meta/OllamaGenerate.png)

### OllamaGenerateAdvance

A node that gives an ability to query an LLM via given prompt with fine tune parameters and an ability to preserve context for generate chaining. 

Check [ollama api docs](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) to get info on the parameters.

More [params info](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter)

![pic](.meta/generate-advance.png)

## Usage Example

Consider the following workflow of vision an image, and perform additional text processing with desired LLM. In the OllamaGenerate node set the prompt as input.

![pic](.meta/CombinedUsage1.png)

The custom Text Nodes in the examples can be found here: https://github.com/pythongosssss/ComfyUI-Custom-Scripts
