package com.automationanywhere.botcommand.utils;

import com.automationanywhere.botcommand.exception.BotCommandException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Shared helpers for action execute() methods.
 */
public class ActionUtils {

    private static final Logger logger = LogManager.getLogger(ActionUtils.class);

    /**
     * Resolve a model ID string to a ModelType, throwing a user-friendly BotCommandException on failure.
     */
    public static ModelManager.ModelType resolveModelType(String modelName) {
        try {
            return ModelManager.ModelType.fromId(modelName.toLowerCase().trim());
        } catch (IllegalArgumentException e) {
            logger.error("Invalid model name: {}", modelName);
            throw new BotCommandException("Invalid model name: " + modelName +
                ". Valid options: " + ModelManager.ModelType.supportedModelIds());
        }
    }

    /**
     * Convert an action's optional "Server Port" field to the Integer override
     * LlamaInference/LlamaServerManager expect. Null, non-positive, or invalid
     * values all mean "no override" — the default automatic port behavior.
     */
    public static Integer resolveServerPort(Double serverPort) {
        if (serverPort == null || serverPort <= 0) {
            return null;
        }
        int port = serverPort.intValue();
        if (port < 1 || port > 65535) {
            logger.warn("Server Port value {} is out of range (1-65535) — using automatic port instead", port);
            return null;
        }
        return port;
    }
}
