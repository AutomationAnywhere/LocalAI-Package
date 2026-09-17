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

import java.util.LinkedHashMap;

import static com.automationanywhere.commandsdk.model.DataType.DICTIONARY;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

/**
 * ClassifyText Action
 *
 * Categorizes text into predefined categories using on-device Small Language Models.
 * Perfect for:
 * - Email triage: "urgent", "normal", "spam"
 * - Document routing: "invoice", "receipt", "contract"
 * - Sentiment analysis: "positive", "negative", "neutral"
 * - Priority classification: "high", "medium", "low"
 *
 * Returns a Dictionary with keys:
 * - category: the matched category name
 * - confidence: score 0.0-1.0 (if Include Confidence is enabled)
 * - explanation: brief reason (if Include Explanation is enabled)
 * - status: "success" or "error"
 * - message: timing and model info
 *
 * Data never leaves your device. No API keys. No cloud calls.
 *
 * First execution downloads model (~600MB-5GB) and may take 10-30 seconds for loading.
 * Subsequent calls are faster (<5 seconds) as models stay in memory.
 */
@BotCommand
@CommandPkg(
    label = "Classify Text",
    name = "classifyText",
    description = "Categorizes text into predefined categories using on-device Small Language Model",
    node_label = "Classify: {{inputText}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#E8F5E9",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class ClassifyText {

    private static final Logger logger = LogManager.getLogger(ClassifyText.class);

    // Max tokens needed for: category name + optional confidence score + optional brief explanation.
    // Capped at the model's own limit via Math.min at call time.
    static final int MAX_OUTPUT_TOKENS = 150;

    // Low temperature for deterministic, consistent classification results.
    static final float CLASSIFICATION_TEMPERATURE = 0.1f;

    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input Text", description = "Text to classify")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.TEXT)
        @Pkg(label = "Categories", description = "Comma-separated list of categories (e.g., 'urgent, normal, spam')")
        @NotEmpty
        String categories,

        // Note: gemma3-270m is intentionally NOT offered here. Tested against 4
        // categories (invoice/receipt/contract/report) across repeated trials, it
        // defaulted to "Receipt" as a catch-all answer — 0/5 correct on contract
        // and report, 2/5 on invoice. Reliable for Prompt (see Prompt.java) but
        // not for multi-category classification at this parameter count.
        @Idx(index = "3", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "3.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "3.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "3.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "3.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "3.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "3.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "3.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b"))
        })
        @Pkg(label = "Model", description = "Small Language Model to use for classification (curated top models under 3GB).", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName,

        @Idx(index = "4", type = AttributeType.CHECKBOX)
        @Pkg(label = "Include Confidence Score", description = "Add 'confidence' key to output (0.0-1.0)", default_value = "false", default_value_type = DataType.BOOLEAN)
        Boolean includeConfidence,

        @Idx(index = "5", type = AttributeType.CHECKBOX)
        @Pkg(label = "Include Explanation", description = "Add 'explanation' key to output with brief reasoning", default_value = "false", default_value_type = DataType.BOOLEAN)
        Boolean includeExplanation,

        @Idx(index = "6", type = AttributeType.NUMBER)
        @Pkg(label = "Timeout (seconds)", description = "Maximum time to wait for processing", default_value = "30", default_value_type = DataType.NUMBER)
        @NotEmpty
        Double timeoutSeconds

    ) {

        logger.info("ClassifyText action started - Model: {}, Categories: {}, Timeout: {}s",
                    modelName, categories, timeoutSeconds);

        try {
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input text cannot be empty");
            }

            if (categories == null || categories.trim().isEmpty()) {
                throw new BotCommandException("Categories cannot be empty");
            }

            if (timeoutSeconds == null || timeoutSeconds <= 0) {
                timeoutSeconds = 30.0;
            }

            if (includeConfidence == null) {
                includeConfidence = false;
            }

            if (includeExplanation == null) {
                includeExplanation = false;
            }

            ModelManager.ModelType modelType = ActionUtils.resolveModelType(modelName);

            long startTime = System.currentTimeMillis();
            LlamaInference inference = new LlamaInference(modelType);
            long loadTime = System.currentTimeMillis() - startTime;
            if (loadTime > 1000) {
                logger.info("Model loaded in {}ms (first-time loading takes longer)", loadTime);
            }

            String prompt = buildClassificationPrompt(inputText, categories, includeConfidence, includeExplanation);
            logger.debug("Generated prompt: {}", prompt.length() > 200 ? prompt.substring(0, 200) + "..." : prompt);

            int effectiveMaxTokens = Math.min(MAX_OUTPUT_TOKENS, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                CLASSIFICATION_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            LinkedHashMap<String, Value<?>> resultFields = parseClassificationResponse(
                rawResponse, categories, includeConfidence, includeExplanation);

            long totalTime = System.currentTimeMillis() - startTime;
            String category = ((StringValue) resultFields.get("category")).get();
            logger.info("Classification completed in {}ms. Result: {}", totalTime, category);

            return DictionaryHelper.success(resultFields, modelType.getId(), totalTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("ClassifyText action failed", e);
            String errorMsg;
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                errorMsg = "Classification timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage() != null && (e.getMessage().contains("model") || e.getMessage().contains("GGUF"))) {
                errorMsg = "Model loading/inference failed: " + e.getMessage() +
                    ". Ensure sufficient memory (~8GB RAM) and model is downloaded.";
            } else {
                errorMsg = "Classification failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            }
            logger.error("Error details: {}", errorMsg);
            throw new BotCommandException(errorMsg, e);
        }
    }

    private String buildClassificationPrompt(String inputText, String categories,
                                             boolean includeConfidence, boolean includeExplanation) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Classify the following text into ONE of these categories: ");
        prompt.append(categories);
        prompt.append("\n\nText to classify: ");
        prompt.append(inputText);
        prompt.append("\n\nRespond with ONLY the category");

        if (includeConfidence) {
            prompt.append(", followed by a pipe symbol and confidence score (0.0-1.0)");
        }
        if (includeExplanation) {
            prompt.append(", followed by a pipe symbol and a brief explanation");
        }
        prompt.append(".\n\n");

        if (includeConfidence && includeExplanation) {
            prompt.append("Example format: category|0.95|brief explanation\n");
        } else if (includeConfidence) {
            prompt.append("Example format: category|0.95\n");
        } else if (includeExplanation) {
            prompt.append("Example format: category|brief explanation\n");
        } else {
            prompt.append("Example format: category\n");
        }
        prompt.append("Classification:");
        return prompt.toString();
    }

    /**
     * Parse the model's classification response into a Dictionary.
     * Package-private for unit testing.
     *
     * @throws BotCommandException if the response is empty or doesn't match any provided category
     */
    LinkedHashMap<String, Value<?>> parseClassificationResponse(String rawResponse, String categories,
                                                                boolean includeConfidence, boolean includeExplanation) {
        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            throw new BotCommandException(
                "Model returned an empty classification response. " +
                "Try increasing the timeout or using a different model.");
        }

        String cleaned = rawResponse.trim();
        cleaned = LlamaInference.stripThinkingBlocks(cleaned);
        cleaned = cleaned.replaceAll("^(Classification:|Category:|Answer:|Result:)\\s*", "");

        if (cleaned.contains("\n")) {
            cleaned = cleaned.substring(0, cleaned.indexOf("\n")).trim();
        }
        cleaned = cleaned.replaceAll("[.!?,;:]+$", "").trim();

        String[] categoryList = categories.split(",");
        String categoryCandidate = cleaned.contains("|") ? cleaned.split("\\|")[0].trim() : cleaned;

        String matchedCategory = null;
        for (String cat : categoryList) {
            if (categoryCandidate.equalsIgnoreCase(cat.trim())) {
                matchedCategory = cat.trim();
                break;
            }
        }

        if (matchedCategory == null) {
            throw new BotCommandException(
                "Model returned '" + categoryCandidate + "' which does not match any expected category. " +
                "Expected one of: " + categories.trim() + ". " +
                "Consider rephrasing the categories or using a more capable model.");
        }

        LinkedHashMap<String, Value<?>> result = new LinkedHashMap<>();
        result.put("category", new StringValue(matchedCategory));

        if (cleaned.contains("|")) {
            String[] parts = cleaned.split("\\|");
            if (includeConfidence && parts.length > 1) {
                result.put("confidence", new StringValue(parts[1].trim()));
            }
            if (includeExplanation) {
                int explanationIdx = includeConfidence ? 2 : 1;
                if (parts.length > explanationIdx) {
                    result.put("explanation", new StringValue(parts[explanationIdx].trim()));
                }
            }
        }

        return result;
    }
}
