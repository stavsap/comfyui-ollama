import { app } from "/scripts/app.js";

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
        let clearMemoryButtonWidget = {};
        if (nodeData.name === "OllamaConnectivityV2") {
          refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");
          clearMemoryButtonWidget = this.addWidget("button", "🗑️ Clear memory");
        }

        const fetchModels = async (url) => {
          const response = await fetch("/ollama/get_models", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              url,
            }),
          });

          if (response.ok) {
            const models = await response.json();
            console.debug("Fetched models:", models);
            return models;
          } else {
            throw new Error(response);
          }
        };

        const updateModels = async () => {
          refreshButtonWidget.name = "⏳ Fetching...";
          const url = urlWidget.value;

          let models = [];
          try {
            models = await fetchModels(url);
          } catch (error) {
            console.error("Error fetching models:", error);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Ollama connection error",
              detail: "Make sure Ollama server is running",
              life: 5000,
            });
            refreshButtonWidget.name = "🔄 Reconnect";
            this.setDirtyCanvas(true);
            return;
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

          refreshButtonWidget.name = "🔄 Reconnect";
          this.setDirtyCanvas(true);
          console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        const clearMemory = async () => {
          clearMemoryButtonWidget.name = "⏳ Clearing...";
          const url = urlWidget.value;
          const model = modelWidget.value;
          const response = await fetch("/ollama/clear_memory", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              url,
              model,
            }),
          });
          if (response.ok) {
            console.debug("Memory cleared");
            app.extensionManager.toast.add({
              severity: "success",
              summary: "Ollama memory cleared",
              detail: "Memory cleared successfully",
              life: 5000,
            });
          } else {
            console.error("Error clearing memory:", response);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Ollama memory clear error",
              detail: response.error,
              life: 5000,
            });
          }
          clearMemoryButtonWidget.name = "🗑️ Clear memory";
        };

        urlWidget.callback = updateModels;
        refreshButtonWidget.callback = updateModels;
        clearMemoryButtonWidget.callback = clearMemory

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
