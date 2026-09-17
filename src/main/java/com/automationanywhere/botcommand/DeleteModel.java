package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.BooleanValue;
import com.automationanywhere.botcommand.exception.BotCommandException;
import com.automationanywhere.botcommand.utils.LlamaServerManager;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.automationanywhere.commandsdk.annotations.*;
import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;
import com.automationanywhere.commandsdk.model.AllowedTarget;
import com.automationanywhere.commandsdk.model.AttributeType;
import com.automationanywhere.commandsdk.model.DataType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import static com.automationanywhere.commandsdk.model.DataType.BOOLEAN;

/**
 * DeleteModel Action
 *
 * Deletes a downloaded Small Language Model from the local storage directory.
 * This action will:
 * 1. Check if the model exists locally
 * 2. Unload the model from memory if currently loaded
 * 3. Delete the model file and directory
 * 4. Return true if deletion was successful
 *
 * Use this to free up disk space when a model is no longer needed.
 * Model storage locations:
 * - Windows: C:\Users\{username}\localAI\
 * - macOS: /Users/{username}/localAI/
 *
 * Note: Model files are large (669MB-5GB), so deletion can free significant space.
 */
@BotCommand
@CommandPkg(
    label = "Delete Model",
    name = "deleteModel",
    description = "Delete a downloaded model from local storage to free disk space",
    node_label = "Delete Model: {{modelName}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#FFFFFF",
    background_color = "#E74C3C",
    allowed_agent_targets = {
//                AllowedTarget.HEADLESS,
            AllowedTarget.WINDOWS,
            AllowedTarget.MAC_OS
//                AllowedTarget.ONDEMAND_CLOUD
    },
    return_type = BOOLEAN,
    return_required = true
)
public class DeleteModel {

    private static final Logger logger = LogManager.getLogger(DeleteModel.class);

    /**
     * Execute model deletion
     *
     * @param modelName The model to delete
     * @return True if model was successfully deleted or didn't exist
     */
    @Execute
    public Value<Boolean> execute(

        @Idx(index = "1", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "1.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "1.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "1.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "1.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "1.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "1.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "1.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b")),
            @Idx.Option(index = "1.8", pkg = @Pkg(label = "Gemma 3 270M (Q4, ~253MB, 32K ctx) — smallest & fastest, narrow tasks only", value = "gemma3-270m"))
        })
        @Pkg(label = "Model", description = "Small Language Model to delete from local storage (curated top models under 3GB).", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName

    ) {

        logger.info("DeleteModel action started - Model: {}", modelName);

        try {
            // Parse model type
            ModelManager.ModelType modelType;
            try {
                modelType = ModelManager.ModelType.fromId(modelName.toLowerCase().trim());
            } catch (IllegalArgumentException e) {
                logger.error("Invalid model name: {}", modelName);
                throw new BotCommandException("Invalid model name: " + modelName +
                    ". Valid options: " + ModelManager.ModelType.supportedModelIds());
            }

            logger.info("Deleting model: {} (~{}MB)", modelType.getId(), modelType.getSizeMB());

            // Get model path and directory
            ModelManager manager = ModelManager.getInstance();
            Path modelPath = manager.getModelPath(modelType);
            Path modelDir = manager.getModelDirectory(modelType);

            logger.debug("Model path: {}", modelPath);
            logger.debug("Model directory: {}", modelDir);

            // Check if model exists
            boolean modelExists = Files.exists(modelPath);

            if (!modelExists) {
                logger.info("Model does not exist at: {}. Nothing to delete.", modelPath);
                return new BooleanValue(true);
            }

            // Get file size before deletion for logging
            long fileSize = Files.size(modelPath);
            logger.info("Found model file: {}MB", fileSize / 1024 / 1024);

            // Stop inference server if it's running this model (server will restart on next use)
            LlamaServerManager server = LlamaServerManager.getInstance();
            if (server.isRunning()) {
                logger.info("Stopping inference server before deleting model...");
                server.stop();
            }

            // Delete the model file
            try {
                Files.delete(modelPath);
                logger.info("Model file deleted: {}", modelPath);
            } catch (Exception e) {
                logger.error("Failed to delete model file", e);
                throw new BotCommandException("Failed to delete model file: " + e.getMessage());
            }

            // Try to delete the directory (will only succeed if empty)
            try {
                if (Files.exists(modelDir) && Files.isDirectory(modelDir)) {
                    try (Stream<Path> dirStream = Files.list(modelDir)) {
                        if (dirStream.findAny().isEmpty()) {
                            Files.delete(modelDir);
                            logger.info("Model directory deleted: {}", modelDir);
                        } else {
                            logger.debug("Model directory not empty, keeping: {}", modelDir);
                        }
                    }
                }
            } catch (Exception e) {
                // Non-critical - directory deletion is best effort
                logger.debug("Could not delete model directory (non-critical): {}", e.getMessage());
            }

            logger.info("Model deletion complete. Freed ~{}MB of disk space", fileSize / 1024 / 1024);

            return new BooleanValue(true);

        } catch (BotCommandException e) {
            // Re-throw bot command exceptions
            throw e;

        } catch (Exception e) {
            logger.error("DeleteModel action failed", e);
            throw new BotCommandException("Model deletion failed: " + e.getMessage(), e);
        }
    }
}
