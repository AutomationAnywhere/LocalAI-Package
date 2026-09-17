package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.exception.BotCommandException;
import com.automationanywhere.botcommand.utils.ActionUtils;
import com.automationanywhere.botcommand.utils.DictionaryHelper;
import com.automationanywhere.botcommand.utils.LlamaInference;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.automationanywhere.commandsdk.annotations.*;
import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;
import com.automationanywhere.commandsdk.model.AllowedTarget;
import com.automationanywhere.commandsdk.model.AttributeType;
import com.automationanywhere.commandsdk.model.DataType;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import static com.automationanywhere.commandsdk.model.DataType.DICTIONARY;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

/**
 * SanitizeJSON Action
 *
 * Uses a local Small Language Model to repair and sanitize malformed or
 * broken JSON — fixing unescaped quotes, bare backslashes, unescaped
 * newlines/tabs, trailing commas, and other common issues.
 *
 * Returns a Dictionary with keys:
 * - sanitized_json: the repaired, valid JSON string
 * - status: "success" or "error"
 * - message: timing and model info
 *
 * First execution downloads the selected model and may take 1-3 minutes.
 * Subsequent calls reuse the cached model in memory.
 */
@BotCommand
@CommandPkg(
    label = "Sanitize JSON",
    name = "sanitizeJSON",
    description = "Repairs malformed JSON (unescaped quotes, backslashes, newlines, trailing commas, etc.) using an on-device Small Language Model",
    node_label = "Sanitize JSON using {{modelName}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#ECF0F1",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class SanitizeJSON {

    private static final Logger logger = LogManager.getLogger(SanitizeJSON.class);
    private static final Gson gson = new Gson();
    private static final Gson prettyGson = new GsonBuilder().setPrettyPrinting().create();

    static final int MAX_OUTPUT_TOKENS = 512;
    static final float JSON_TEMPERATURE = 0.1f;

    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input JSON", description = "The malformed or broken JSON text to sanitize and repair")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "2.1", pkg = @Pkg(label = "Compact (no whitespace)", value = "compact")),
            @Idx.Option(index = "2.2", pkg = @Pkg(label = "Pretty (formatted)", value = "pretty"))
        })
        @Pkg(label = "Output Style", description = "JSON formatting style for the sanitized output", default_value = "compact", default_value_type = DataType.STRING)
        @NotEmpty
        String outputStyle,

        // Note: gemma3-270m is intentionally NOT offered here. In live testing,
        // given JSON with an unescaped nested quote, it returned validly-formed
        // JSON but silently truncated the string value instead of repairing it —
        // data loss disguised as success. Reliable for Prompt only.
        @Idx(index = "3", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "3.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "3.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "3.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "3.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "3.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "3.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "3.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b"))
        })
        @Pkg(label = "Model", description = "Small Language Model to use for JSON repair (curated top models under 3GB).", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName,

        @Idx(index = "4", type = AttributeType.NUMBER)
        @Pkg(label = "Timeout (seconds)", description = "Maximum time to wait for processing", default_value = "30", default_value_type = DataType.NUMBER)
        @NotEmpty
        Double timeoutSeconds

    ) {
        logger.info("SanitizeJSON action started - Model: {}, OutputStyle: {}, Timeout: {}s",
                    modelName, outputStyle, timeoutSeconds);

        try {
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input JSON cannot be empty");
            }

            if (outputStyle == null || outputStyle.trim().isEmpty()) {
                outputStyle = "compact";
            }

            if (timeoutSeconds == null || timeoutSeconds <= 0) {
                timeoutSeconds = 30.0;
            }

            ModelManager.ModelType modelType = ActionUtils.resolveModelType(modelName);

            long startTime = System.currentTimeMillis();
            LlamaInference inference = new LlamaInference(modelType);
            long loadTime = System.currentTimeMillis() - startTime;

            if (loadTime > 1000) {
                logger.info("Model loaded in {}ms (first-time loading takes longer)", loadTime);
            }

            String prompt = buildSanitizePrompt(inputText);
            logger.debug("Prompt length: {} chars", prompt.length());

            int effectiveMaxTokens = Math.min(MAX_OUTPUT_TOKENS, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                JSON_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            String jsonResult = parseAndValidateJSON(rawResponse, outputStyle);

            long totalTime = System.currentTimeMillis() - startTime;
            logger.info("SanitizeJSON completed in {}ms. Output length: {} chars", totalTime, jsonResult.length());

            return DictionaryHelper.success("sanitized_json", jsonResult, modelType.getId(), totalTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("SanitizeJSON action failed", e);

            String errorMsg;
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                errorMsg = "Sanitization timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage() != null && e.getMessage().contains("model")) {
                errorMsg = "Model loading/inference failed: " + e.getMessage();
            } else {
                errorMsg = "Sanitization failed: " + e.getMessage();
            }

            throw new BotCommandException(errorMsg, e);
        }
    }

    /**
     * Build the prompt instructing the model to repair and return valid JSON.
     */
    private String buildSanitizePrompt(String inputText) {
        return "You are a JSON repair tool. Fix the broken JSON below so it is valid.\n" +
               "Rules:\n" +
               "- Escape unescaped double quotes inside string values\n" +
               "- Escape backslashes that are not already escaped\n" +
               "- Replace literal newlines and tabs inside strings with \\n and \\t\n" +
               "- Remove trailing commas\n" +
               "- Do not change key names or values beyond what is needed to make the JSON valid\n" +
               "- Output ONLY the repaired JSON, nothing else\n\n" +
               "Broken JSON:\n" +
               inputText + "\n\n" +
               "Repaired JSON:";
    }

    /**
     * Strip thinking blocks, extract JSON structure, validate with Gson, and format output.
     * Package-private for unit testing.
     */
    String parseAndValidateJSON(String rawResponse, String outputStyle) throws BotCommandException {
        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            throw new BotCommandException("Model returned empty response");
        }

        String cleaned = rawResponse.trim();

        // Strip DeepSeek R1 thinking blocks
        cleaned = LlamaInference.stripThinkingBlocks(cleaned);

        // Remove markdown code block markers
        cleaned = cleaned.replaceAll("^```json\\s*\\n?", "");
        cleaned = cleaned.replaceAll("^```\\s*\\n?", "");
        cleaned = cleaned.replaceAll("\\n?```$", "");
        cleaned = cleaned.trim();

        // Find first { or [
        int jsonStart = -1;
        for (int i = 0; i < cleaned.length(); i++) {
            char c = cleaned.charAt(i);
            if (c == '{' || c == '[') {
                jsonStart = i;
                break;
            }
        }

        if (jsonStart == -1) {
            throw new BotCommandException(
                "Model did not return a JSON structure. Response: " +
                (cleaned.length() > 100 ? cleaned.substring(0, 100) + "..." : cleaned));
        }

        cleaned = cleaned.substring(jsonStart);

        // Find matching closing bracket
        int depth = 0;
        int jsonEnd = -1;
        char startChar = cleaned.charAt(0);
        char endChar = (startChar == '{') ? '}' : ']';

        for (int i = 0; i < cleaned.length(); i++) {
            char c = cleaned.charAt(i);
            if (c == startChar) depth++;
            else if (c == endChar) {
                depth--;
                if (depth == 0) { jsonEnd = i; break; }
            }
        }

        if (jsonEnd != -1) {
            cleaned = cleaned.substring(0, jsonEnd + 1);
        }

        // Validate with Gson
        try {
            Object parsed = gson.fromJson(cleaned, Object.class);
            if ("pretty".equals(outputStyle)) {
                return prettyGson.toJson(parsed);
            } else {
                return gson.toJson(parsed);
            }
        } catch (JsonSyntaxException e) {
            logger.error("Model output is still invalid JSON: {}", cleaned);
            throw new BotCommandException(
                "Model could not produce valid JSON. Output was: " +
                (cleaned.length() > 150 ? cleaned.substring(0, 150) + "..." : cleaned));
        }
    }
}
