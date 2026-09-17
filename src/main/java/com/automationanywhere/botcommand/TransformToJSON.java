package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.exception.BotCommandException;
import com.automationanywhere.botcommand.utils.ActionUtils;
import com.automationanywhere.botcommand.utils.DictionaryHelper;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.automationanywhere.botcommand.utils.LlamaInference;
import com.automationanywhere.commandsdk.annotations.*;
import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;
import com.automationanywhere.commandsdk.model.AllowedTarget;
import com.automationanywhere.commandsdk.model.AttributeType;
import com.automationanywhere.commandsdk.model.DataType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import static com.automationanywhere.commandsdk.model.DataType.DICTIONARY;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

/**
 * TransformToJSON Action
 *
 * Converts various text formats into valid JSON strings using Small Language Models.
 * Perfect for:
 * - CSV data → JSON array for REST APIs
 * - Key-value text → JSON object for logging/storage
 * - Table data → JSON for database operations
 * - Unstructured text → structured JSON
 *
 * Returns a Dictionary with keys:
 * - json: a valid JSON string (either object {} or array [])
 * - status: "success" or "error"
 * - message: timing and model info
 *
 * First execution downloads model (~600MB-5GB) and may take 10-30 seconds for loading.
 * Subsequent calls are faster (<5 seconds) as models stay in memory.
 */
@BotCommand
@CommandPkg(
    label = "Transform To JSON",
    name = "transformToJSON",
    description = "Converts various text formats into valid JSON strings using on-device Small Language Model",
    node_label = "Transform to JSON: {{inputText}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#FFF3E0",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class TransformToJSON {

    private static final Logger logger = LogManager.getLogger(TransformToJSON.class);
    private static final Gson gson = new Gson();

    // Max tokens for JSON output — JSON can be verbose, especially for multi-row arrays.
    // Capped at the model's own limit via Math.min at call time.
    static final int MAX_OUTPUT_TOKENS = 300;

    // Low temperature for consistent, structured JSON output.
    static final float JSON_TEMPERATURE = 0.2f;

    /**
     * Execute text to JSON transformation using a Small Language Model
     *
     * @param inputText The text to transform into JSON
     * @param inputFormat Format of input: "CSV", "TSV", "Key-Value", "Table", "Auto-detect"
     * @param outputStyle JSON output style: "compact" (no whitespace) or "pretty" (formatted)
     * @param outputType Output structure: "object" (single {}) or "array" (multiple [{}])
     * @param modelName The model to use (qwen3-4b, llama3.2-3b, phi4-mini, gemma3-4b)
     * @param timeoutSeconds Maximum time to wait for processing (default: 30)
     * @return Valid JSON string
     */
    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input Text", description = "Text to transform into JSON")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "2.1", pkg = @Pkg(label = "CSV", value = "csv")),
            @Idx.Option(index = "2.2", pkg = @Pkg(label = "TSV (Tab-Separated)", value = "tsv")),
            @Idx.Option(index = "2.3", pkg = @Pkg(label = "Key-Value Pairs", value = "key-value")),
            @Idx.Option(index = "2.4", pkg = @Pkg(label = "Table", value = "table")),
            @Idx.Option(index = "2.5", pkg = @Pkg(label = "List/Bullet Points", value = "list")),
            @Idx.Option(index = "2.6", pkg = @Pkg(label = "Auto-Detect", value = "auto-detect"))
        })
        @Pkg(label = "Input Format", description = "Format of the input text", default_value = "auto-detect", default_value_type = DataType.STRING)
        @NotEmpty
        String inputFormat,

        @Idx(index = "3", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "3.1", pkg = @Pkg(label = "Compact (no whitespace)", value = "compact")),
            @Idx.Option(index = "3.2", pkg = @Pkg(label = "Pretty (formatted)", value = "pretty"))
        })
        @Pkg(label = "Output Style", description = "JSON formatting style", default_value = "compact", default_value_type = DataType.STRING)
        @NotEmpty
        String outputStyle,

        @Idx(index = "4", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "4.1", pkg = @Pkg(label = "Object (single {})", value = "object")),
            @Idx.Option(index = "4.2", pkg = @Pkg(label = "Array (multiple [{}])", value = "array"))
        })
        @Pkg(label = "Output Type", description = "JSON structure type", default_value = "object", default_value_type = DataType.STRING)
        @NotEmpty
        String outputType,

        // Note: gemma3-270m is intentionally NOT offered here. In live testing it
        // returned syntactically valid JSON with fabricated content instead of
        // extracted content — e.g. given a 2-row CSV it invented wrong ages and
        // dropped a column entirely; given key-value input it dropped real fields
        // and added one that was never in the input. Worse than an obvious
        // failure, since the output looks plausible. Reliable for Prompt only.
        @Idx(index = "5", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "5.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "5.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "5.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "5.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "5.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "5.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "5.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b")),
            @Idx.Option(index = "5.8", pkg = @Pkg(label = "Gemma 3 1B (Q4, ~769MB, 32K ctx) — small & fast, reliable across tasks", value = "gemma3-1b"))
        })
        @Pkg(label = "Model", description = "Small Language Model to use for transformation (curated top models under 3GB).", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName,

        @Idx(index = "6", type = AttributeType.NUMBER)
        @Pkg(label = "Timeout (seconds)", description = "Maximum time to wait for processing", default_value = "30", default_value_type = DataType.NUMBER)
        @NotEmpty
        Double timeoutSeconds

    ) {

        logger.info("TransformToJSON action started - Model: {}, Format: {}, Style: {}, Type: {}, Timeout: {}s",
                    modelName, inputFormat, outputStyle, outputType, timeoutSeconds);

        try {
            // Validate inputs
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input text cannot be empty");
            }

            if (inputFormat == null || inputFormat.trim().isEmpty()) {
                inputFormat = "auto-detect";
            }

            if (outputStyle == null || outputStyle.trim().isEmpty()) {
                outputStyle = "compact";
            }

            if (outputType == null || outputType.trim().isEmpty()) {
                outputType = "object";
            }

            if (timeoutSeconds == null || timeoutSeconds <= 0) {
                timeoutSeconds = 30.0;
            }

            ModelManager.ModelType modelType = ActionUtils.resolveModelType(modelName);

            logger.debug("Using model: {} for text length: {}", modelType.getId(), inputText.length());

            // Initialize inference engine (will load model if needed)
            long startTime = System.currentTimeMillis();
            LlamaInference inference = new LlamaInference(modelType);
            long loadTime = System.currentTimeMillis() - startTime;

            if (loadTime > 1000) {
                logger.info("Model loaded in {}ms (first-time loading takes longer)", loadTime);
            }

            // Build transformation prompt
            String prompt = buildTransformationPrompt(inputText, inputFormat, outputType);
            logger.debug("Generated prompt: {}", prompt.length() > 200 ? prompt.substring(0, 200) + "..." : prompt);

            // Generate JSON.
            //
            // NOTE on grammar-constrained generation: JsonGrammar provides GBNF grammars
            // that can be passed to llama-server to guarantee syntactically valid JSON at
            // the token-sampling level.  However, grammar constraints work best when the
            // prompt is "primed" — i.e. the opening '{' or '[' is already appended to the
            // prompt so the model generates content from *inside* the structure rather than
            // choosing the minimal valid path ({} or []).  The current chat-template prompt
            // structure is not set up for that priming, so grammar is not applied here.
            // When a primed-prompt path is added, wire in: JsonGrammar.forOutputType(outputType)
            int effectiveMaxTokens = Math.min(MAX_OUTPUT_TOKENS, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                JSON_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            // Parse, validate, and format the JSON response
            String jsonResult = parseAndValidateJSON(rawResponse, outputStyle, outputType);

            long totalTime = System.currentTimeMillis() - startTime;
            logger.info("Transformation completed in {}ms. Output length: {} chars", totalTime, jsonResult.length());

            return DictionaryHelper.success("json", jsonResult, modelType.getId(), totalTime);

        } catch (Exception e) {
            logger.error("TransformToJSON action failed", e);

            // Provide helpful error messages
            String errorMsg;
            if (e.getMessage().contains("timeout")) {
                errorMsg = "Transformation timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage().contains("model") || e.getMessage().contains("GGUF")) {
                errorMsg = "Model loading/inference failed: " + e.getMessage() +
                    ". Ensure sufficient memory (~8GB RAM) and model is downloaded.";
            } else if (e.getMessage().contains("JSON")) {
                errorMsg = "Failed to generate valid JSON: " + e.getMessage() + ". Try using a larger model (Gemma 2B or Qwen2.5 3B).";
            } else {
                errorMsg = "Transformation failed: " + e.getMessage();
            }

            logger.error("Error details: {}", errorMsg);
            throw new BotCommandException(errorMsg, e);
        }
    }

    /**
     * Build the transformation prompt based on input format and output type
     */
    private String buildTransformationPrompt(String inputText, String inputFormat, String outputType) {
        StringBuilder prompt = new StringBuilder();

        // Use a more direct instruction format that works better with small models
        prompt.append("Task: Convert text to JSON\n\n");

        // Add format-specific instructions with clear examples
        switch (inputFormat.toLowerCase().trim()) {
            case "csv":
                prompt.append("Format: CSV\n");
                if (outputType.equals("array")) {
                    prompt.append("Example:\nInput: Name,Age\nJohn,30\nJane,25\nOutput: [{\"Name\":\"John\",\"Age\":\"30\"},{\"Name\":\"Jane\",\"Age\":\"25\"}]\n\n");
                } else {
                    prompt.append("Example:\nInput: Name,Age\nJohn,30\nOutput: {\"Name\":\"John\",\"Age\":\"30\"}\n\n");
                }
                break;

            case "tsv":
                prompt.append("Format: TSV (tab-separated)\n");
                if (outputType.equals("array")) {
                    prompt.append("Output: JSON array\n\n");
                } else {
                    prompt.append("Output: JSON object\n\n");
                }
                break;

            case "key-value":
                prompt.append("Format: Key-Value pairs\n");
                prompt.append("Example:\nInput: Name: John\nAge: 30\nOutput: {\"Name\":\"John\",\"Age\":\"30\"}\n\n");
                break;

            case "table":
                prompt.append("Format: Text table\n");
                if (outputType.equals("array")) {
                    prompt.append("Output: JSON array\n\n");
                } else {
                    prompt.append("Output: JSON object\n\n");
                }
                break;

            case "list":
                prompt.append("Format: List\n");
                if (outputType.equals("array")) {
                    prompt.append("Example:\nInput: - Apple\n- Banana\nOutput: [\"Apple\",\"Banana\"]\n\n");
                } else {
                    prompt.append("Example:\nInput: - Apple\n- Banana\nOutput: {\"item1\":\"Apple\",\"item2\":\"Banana\"}\n\n");
                }
                break;

            case "auto-detect":
            default:
                prompt.append("Format: Auto-detect\n");
                if (outputType.equals("array")) {
                    prompt.append("Output: JSON array []\n\n");
                } else {
                    prompt.append("Output: JSON object {}\n\n");
                }
                break;
        }

        prompt.append("Input:\n").append(inputText).append("\n\n");
        prompt.append("Output:");

        return prompt.toString();
    }

    /**
     * Validate and format the JSON response.
     * Package-private for unit testing.
     *
     * With grammar-constrained generation the model output is guaranteed to be
     * syntactically valid JSON, so the heavy cleanup heuristics (strip markdown,
     * find first '{', bracket-match) are no longer the primary path.  They are
     * retained as a defensive fallback for non-grammar callers or future refactors.
     *
     * @throws BotCommandException if the response is empty, contains no JSON, or the
     *         JSON structure does not match the requested outputType
     */
    String parseAndValidateJSON(String rawResponse, String outputStyle, String outputType)
            throws BotCommandException {
        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            throw new BotCommandException("Model returned empty response");
        }

        String cleaned = rawResponse.trim();

        // --- Defensive cleanup (no-op when grammar was applied) ---

        // Strip DeepSeek R1 thinking blocks (<think>...</think>). Grammar prevents
        // these, but DeepSeek may still emit partial blocks before the JSON starts
        // on some builds.
        cleaned = LlamaInference.stripThinkingBlocks(cleaned);

        // Remove markdown fences — grammar prevents these, kept for safety.
        cleaned = cleaned.replaceAll("^```json\\s*\\n?", "");
        cleaned = cleaned.replaceAll("^```\\s*\\n?", "");
        cleaned = cleaned.replaceAll("\\n?```$", "");
        cleaned = cleaned.trim();

        // Advance past any preamble to the first '{' or '['.
        int jsonStart = -1;
        for (int i = 0; i < cleaned.length(); i++) {
            char c = cleaned.charAt(i);
            if (c == '{' || c == '[') { jsonStart = i; break; }
        }

        if (jsonStart == -1) {
            logger.error("No JSON structure found in response. Response preview: {}",
                cleaned.length() > 200 ? cleaned.substring(0, 200) : cleaned);
            throw new BotCommandException(
                "No JSON structure found in model response. Response: " +
                (cleaned.length() > 100 ? cleaned.substring(0, 100) + "..." : cleaned));
        }
        cleaned = cleaned.substring(jsonStart);

        // --- Validate and format ---

        try {
            Object parsed = gson.fromJson(cleaned, Object.class);

            boolean isArray  = parsed instanceof java.util.List;
            boolean isObject = parsed instanceof java.util.Map;

            // Reject semantically empty results — {} or [] mean the model produced
            // no useful transformation.  This most commonly happens when grammar-
            // constrained generation is applied without prompt priming: the model
            // takes the minimal-valid-JSON path instead of filling in the data.
            if (isObject && ((java.util.Map<?,?>) parsed).isEmpty()) {
                throw new BotCommandException(
                    "Model returned empty JSON object {}. The transformation produced no data. " +
                    "Try re-running, using a different model, or increasing the timeout.");
            }
            if (isArray && ((java.util.List<?>) parsed).isEmpty()) {
                throw new BotCommandException(
                    "Model returned empty JSON array []. The transformation produced no data. " +
                    "Try re-running, using a different model, or increasing the timeout.");
            }

            if (outputType.equals("array") && !isArray) {
                throw new BotCommandException(
                    "Model generated a JSON object but an array was requested. " +
                    "Change the Output Type to 'Object', or adjust the input so it contains multiple rows.");
            } else if (outputType.equals("object") && !isObject) {
                throw new BotCommandException(
                    "Model generated a JSON array but an object was requested. " +
                    "Change the Output Type to 'Array', or adjust the input so it contains a single record.");
            }

            if (outputStyle.equals("pretty")) {
                Gson prettyGson = new com.google.gson.GsonBuilder().setPrettyPrinting().create();
                return prettyGson.toJson(parsed);
            }
            return gson.toJson(parsed);

        } catch (JsonSyntaxException e) {
            logger.error("Invalid JSON in model response: {}", cleaned);
            throw new BotCommandException(
                "Generated invalid JSON: " + e.getMessage() +
                "\nGenerated content: " + cleaned);
        }
    }
}
