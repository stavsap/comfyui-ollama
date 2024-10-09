09-10-2024:

Add out put format `text` or `json` in all generate nodes.

(should be compatible with existing workflows)

31-08-2024:

Merge change `top_k` and `num_predict` params in **OllamaAdvanced** to be INT instead FLOAT.

30-07-2024

Add 2 new Nodes:

- OllamaSaveContext: will save context from OllamaGenerateAdvance to a `png` file in saved_context folder.
- OllamaLoadContext: will load context for OllamaGenerateAdvance from a `png` files in saved_context folder.
- `context` input in OllamaGenerateAdvance node now can handle string also.

18-06-2024

- Add `keep_context` option in OllamaGenerateAdvance, this flag will preserve context from the current run in the node itself.
- Modify debug flag to be boolean.

24-05-2024

- Add drop down list to select ollama models that currently available in ollama server.

23-05-2024

- Add `keep_alive` option. Controls for how long the selected model will stay loaded into memory following the request (default: 5m).

21-05-2024

- Modify Ollama Generate Advance node, change seed to 'INT' from 'FLOAT'.
