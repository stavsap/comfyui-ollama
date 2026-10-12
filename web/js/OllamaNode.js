import { app } from "/scripts/app.js";

const postJson = async (path, body) => {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(data?.error || `${response.status} ${response.statusText}`);
    error.summary = data?.summary; // set for setup problems, such as an outdated ollama package
    throw error;
  }
  return data;
};

// Setup problems stay on screen until dismissed, so the fix can be read and copied.
const showError = (summary, detail, error) => {
  app.extensionManager.toast.add(
    error.summary
      ? { severity: "error", summary: error.summary, detail: error.message }
      : { severity: "error", summary, detail, life: 5000 },
  );
};

app.registerExtension({
  name: "Comfy.OllamaNode",
  aboutPageBadges: [
    {
      label: "ComfyUI-Ollama",
      url: "https://github.com/stavsap/comfyui-ollama",
      icon: "pi pi-github",
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaGenerate", "OllamaGenerateAdvance", "OllamaVision", "OllamaConnectivityV2"].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");
        let refreshButtonWidget = {};
        let unloadButtonWidget = null;
        if (nodeData.name === "OllamaConnectivityV2") {
          refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");
          // Not saved in workflows or prompts, so saved widget values keep their positions.
          unloadButtonWidget = this.addWidget("button", "🧹 Unload models", null, null, { serialize: false });
          unloadButtonWidget.serialize = false;
        }

        const fetchModels = async (url) => {
          const models = await postJson("/ollama/get_models", { url });
          console.debug("Fetched models:", models);
          return models;
        };

        let fetching = false;
        const updateModels = async () => {
          if (fetching) {
            return;
          }
          fetching = true;
          refreshButtonWidget.name = "⏳ Fetching...";
          const url = urlWidget.value;

          let models = [];
          try {
            models = await fetchModels(url);
          } catch (error) {
            console.error("Error fetching models:", error);
            showError("Ollama connection error", `Could not load the model list from ${url}: ${error.message}`, error);
            return;
          } finally {
            fetching = false;
            refreshButtonWidget.name = "🔄 Reconnect";
            this.setDirtyCanvas(true);
          }

          const prevValue = modelWidget.value;

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // stay on current.
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // set first as default.
          }

          this.setDirtyCanvas(true);
          console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        // Unload every model Ollama has in memory, so ComfyUI gets the VRAM back.
        const unloadModels = async () => {
          unloadButtonWidget.name = "⏳ Unloading...";
          this.setDirtyCanvas(true);
          try {
            const unloaded = await postJson("/ollama/unload", { url: urlWidget.value });
            app.extensionManager.toast.add({
              severity: "success",
              summary: "Ollama models unloaded",
              detail: unloaded.length > 0 ? unloaded.join(", ") : "No models were loaded",
              life: 4000,
            });
          } catch (error) {
            console.error("Error unloading models:", error);
            showError("Ollama unload failed", error.message, error);
          } finally {
            unloadButtonWidget.name = "🧹 Unload models";
            this.setDirtyCanvas(true);
          }
        };

        urlWidget.callback = updateModels;
        refreshButtonWidget.callback = updateModels;
        if (unloadButtonWidget) {
          unloadButtonWidget.callback = unloadModels;
        }

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        };

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();
      };
    }
  },
});
