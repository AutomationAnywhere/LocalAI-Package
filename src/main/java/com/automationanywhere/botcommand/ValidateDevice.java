package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.exception.BotCommandException;
import com.automationanywhere.botcommand.utils.DictionaryHelper;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.automationanywhere.botcommand.utils.ModelDownloader;
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
import java.util.LinkedHashMap;

import static com.automationanywhere.commandsdk.model.DataType.DICTIONARY;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

/**
 * ValidateDevice Action
 *
 * Downloads or validates that a Small Language Model is available on the device.
 * Use this action before running inference to ensure the model is ready.
 *
 * This action will:
 * 1. Check if the model already exists locally
 * 2. If not, download it from HuggingFace (~669MB-5GB depending on model)
 * 3. Validate the model file is complete and accessible
 * 4. Return a Dictionary with model details and ready status
 *
 * Model storage locations:
 * - Windows: %LOCALAPPDATA%\AutomationAnywhere\LocalAI\
 * - macOS:   ~/localAI/
 *
 * Also downloads the llama.cpp server binary (~9-15MB) on first use.
 * First-time model download may take 5-15 minutes depending on internet speed.
 *
 * Return keys: supported, model_name, model_path, model_size_mb, status, message
 */
@BotCommand
@CommandPkg(
    label = "Validate Device",
    name = "validateDevice",
    description = "Download or validate that a model is available. Windows: %LOCALAPPDATA%\\AutomationAnywhere\\LocalAI\\, macOS: ~/localAI/",
    node_label = "Validate Device: {{modelName}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#3498DB",
    allowed_agent_targets = {
            AllowedTarget.WINDOWS,
            AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class ValidateDevice {

    private static final Logger logger = LogManager.getLogger(ValidateDevice.class);

    @Execute
    public DictionaryValue execute(

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
        @Pkg(label = "Model", description = "Small Language Model to validate/download (curated top models under 3GB).", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName

    ) {

        logger.info("ValidateDevice action started - Model: {}", modelName);

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

            long startTime = System.currentTimeMillis();
            logger.info("Validating model: {} (~{}MB)", modelType.getId(), modelType.getSizeMB());

            // Ensure the llama-server binary is installed (downloads ~9-15MB on first use)
            try {
                ModelDownloader.ensureLlamaBinary();
            } catch (Exception e) {
                logger.error("Failed to install llama-server binary", e);
                throw new BotCommandException("Failed to install llama-server: " + e.getMessage() +
                    ". Please check internet connection.");
            }

            ModelManager manager = ModelManager.getInstance();
            Path modelPath = manager.getModelPath(modelType);

            logger.debug("Model path: {}", modelPath);

            boolean modelExists = Files.exists(modelPath);

            if (modelExists) {
                logger.info("Model already exists at: {}", modelPath);

                try {
                    long fileSize = Files.size(modelPath);
                    // Require at least 90% of expected size to consider the file complete
                    long expectedMinSize = (long) modelType.getSizeMB() * 1024 * 1024 * 9 / 10;

                    if (fileSize < expectedMinSize) {
                        logger.warn("Model file exists but seems incomplete ({}MB, expected ~{}MB). Re-downloading...",
                            fileSize / 1024 / 1024, modelType.getSizeMB());
                        Files.delete(modelPath);
                        modelExists = false;
                    } else {
                        logger.info("Model file validated: {}MB", fileSize / 1024 / 1024);
                    }

                } catch (Exception e) {
                    logger.error("Error validating existing model file", e);
                    throw new BotCommandException("Error validating model file: " + e.getMessage());
                }
            }

            if (!modelExists) {
                logger.info("Model not found. Starting download (~{}MB)...", modelType.getSizeMB());
                logger.info("This may take 5-15 minutes depending on internet speed.");

                try {
                    ModelDownloader.downloadModel(modelType);
                    logger.info("Model downloaded successfully to: {}", modelPath);
                } catch (Exception e) {
                    logger.error("Model download failed", e);
                    throw new BotCommandException("Failed to download model: " + e.getMessage() +
                        ". Please check internet connection and disk space.");
                }
            }

            if (!Files.exists(modelPath)) {
                throw new BotCommandException("Model validation failed: File does not exist after download");
            }

            long finalSize = Files.size(modelPath);
            logger.info("Device validation complete. Model ready at: {} ({}MB)", modelPath, finalSize / 1024 / 1024);

            LinkedHashMap<String, Value<?>> fields = new LinkedHashMap<>();
            fields.put("supported", new StringValue("true"));
            fields.put("model_name", new StringValue(modelType.getId()));
            fields.put("model_path", new StringValue(modelPath.toString()));
            fields.put("model_size_mb", new StringValue(String.valueOf(finalSize / 1024 / 1024)));
            return DictionaryHelper.success(fields, modelType.getId(), System.currentTimeMillis() - startTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("ValidateDevice action failed", e);
            throw new BotCommandException("Device validation failed: " + e.getMessage(), e);
        }
    }
}
