import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.OllamaNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaGenerate", "OllamaGenerateAdvance", "OllamaVision"].includes(nodeData.name) ) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const fetchModels = async (url) => {
          try {
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
              console.log("Fetched models:", models);
              return models;
            } else {
              console.error(`Failed to fetch models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching models`, error);
            return [];
          }
        };

        const updateModels = async () => {
          const url = urlWidget.value;

          const models = await fetchModels(url);

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.log("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(modelWidget.value)) {
            modelWidget.value = modelWidget.value;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          } else {
            modelWidget.value = "";
          }
          console.log("Updated modelWidget.value:", modelWidget.value);

          this.triggerSlot(0);
        };

        urlWidget.callback = updateModels;
        // Initial update
        await updateModels();
      };
    }
  },
});
