12-10-2026:

Reliability and speed fixes for current Ollama (tested with Ollama 0.40.3 and ComfyUI 0.39.2).

- Ollama Chat now sends the `think` setting. Many current models think by default, so before this the node always ran a full reasoning pass even with `think` off, which made it much slower.
- Ollama Chat is now stateless by default: each run starts a new conversation. Turn on the new `remember_history` input to keep the conversation across runs. Before this, every run added to the history until ComfyUI restarted, so each run got slower. The `history` output now carries the conversation itself, so chained Chat nodes keep working.
- Responses are streamed. The text appears on the node while it's generated, and ComfyUI's cancel button now stops the generation in Ollama.
- Progress on the node: while waiting, it says whether Ollama is connecting, loading the model or reading the prompt, with the seconds elapsed. While generating, it shows the phase (thinking or writing the answer), the token count and the elapsed time.
- Each run logs a timing summary to the console: model load time, prompt reading, tokens generated, speed, and how many tokens went to thinking.
- Requests time out instead of hanging forever. A wrong URL fails within about 10 seconds, and a request fails if Ollama sends nothing for `request_timeout` seconds (new input on Ollama Options, default 300, 0 = no limit).
- Loading the model list no longer freezes the ComfyUI interface when the Ollama URL is wrong or slow, and errors are shown in the notification.
- The model list is sorted alphabetically.
- New `think_level` input (auto/low/medium/high) on Ollama Generate and Ollama Chat, for models with thinking levels such as gpt-oss.
- If a model writes its reasoning inside `<think>` tags that Ollama doesn't recognize (usually a custom import), Ollama Generate and Ollama Chat move it from `result` to `thinking`.
- New `seed` input on Ollama Generate and Ollama Chat, with ComfyUI's usual control after generate. `fixed` (the default) reuses the previous result while nothing changes; `randomize` gives a fresh answer on every run. The seed on Ollama Options is deprecated; if it's enabled, it still overrides the node's seed.
- Ollama Options: `stop` works again. It was sent as plain text, which current Ollama rejects, so every request failed when it was enabled.
- Ollama Options: `mirostat`, `mirostat_eta`, `mirostat_tau` and `tfs_z` are no longer supported by Ollama. They're kept so older workflows load, but they're no longer sent, and a warning is logged if they're enabled.
- Ollama Options: new `num_gpu`, `presence_penalty` and `frequency_penalty` settings. `num_predict` and `repeat_last_n` accept larger values, and `num_ctx` starts at 8192 when enabled.
- Ollama Options: the deprecated seed now defaults to `fixed`. When it was `randomize`, every connected node re-ran on every queue.
- New "Unload models" button on Ollama Connectivity frees the VRAM Ollama is using, so ComfyUI can use it.
- Fixed options being ignored on chained nodes connected through `meta`.
- Fixed the KeyError 'context' crash in Ollama Generate.
- Requires `ollama>=0.6.2` (the python client), and the unused `dotenv` requirement is removed. If an older version is installed, the nodes say so and show the command to update it.
- Node titles now start with 🦙.

23-10-2025:

- Fixed Ollama cloud model support with new OllamaChat node
- Cloud models authenticate via SSH keys (no API key required)
- Uses `ollama.chat()` method for cloud compatibility

04-08-2025:

- Added `think` option and separate `thinking` output on the Ollama Generate node. This replaces the previous `filter_thinking` workaround.
- Added tooltips and node descriptions.
- Added example workflows. See ComfyUI's template browser in the workflow menu.
- Added a message that appears when the list of models fails to load. User can click the "Reconnect" button on the connectivity node to reload list.
- Deprecate V1 nodes, please replace them in your workflows.

10-07-2025:

Adding user seed to OllamaVision request.

10-06-2025:

Increased temperature limits in OptionsV2.

26-05-2025:

Added toggle to strip thinking from response

22-04-2025:

Fix v2 settings num ctx limit.
Update ollama python 0.4.8.

10-01-2025:

V2 release.

Added nodes:

- OllamaGenerateV2

- OllamaConnectivityV2

- OllamaOptionsV2

09-01-2025

Update python ollama client 0.4.2 -> 0.4.5

04-12-2024:

Fix image base64 to binary issue.

25-11-2024:

Fix some bug in fetching model names.

20-11-2024:

Support step of 1 in `keep_alive`.

16-11-2024:

Support `-1` value in `keep_alive`.

19-10-2024:

Add seed input to OllamaVision node.

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
