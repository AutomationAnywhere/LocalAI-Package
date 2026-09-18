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
 * ExtractData Action
 *
 * Pulls specific fields from unstructured text using on-device AI — no regex, no training.
 * Perfect for:
 * - Invoices: extract invoice_number, total_amount, due_date, vendor_name
 * - Emails: extract sender_intent, urgency, action_required
 * - Support tickets: extract issue_type, customer_id, product_version
 * - Contracts: extract parties, effective_date, value, renewal_terms
 *
 * Define the fields you want in a simple "fieldName: description" format.
 * Returns a Dictionary where each key is a field name — use {invoice_number},
 * {total_amount}, etc. directly in downstream bot actions.
 *
 * Data never leaves your device. No API keys. No cloud calls.
 *
 * First execution downloads model (~600MB-5GB) and may take 10-30 seconds for loading.
 * Subsequent calls are faster (<5 seconds) as models stay in memory.
 */
@BotCommand
@CommandPkg(
    label = "Extract Data",
    name = "extractData",
    description = "Pulls specific fields from unstructured text using on-device AI. Define fields as 'fieldName: description' — get back a Dictionary you can use directly.",
    node_label = "Extract fields from: {{inputText}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#EDE7F6",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class ExtractData {

    private static final Logger logger = LogManager.getLogger(ExtractData.class);

    // Extraction output can span many fields — generous budget, capped per model
    static final int MAX_OUTPUT_TOKENS = 400;

    // Low temperature for consistent, deterministic field extraction
    static final float EXTRACT_TEMPERATURE = 0.1f;

    static final String NOT_FOUND = "NOT_FOUND";

    /**
     * Extract structured fields from unstructured text.
     *
     * @param inputText      The document, email, or text to extract from
     * @param fieldsToExtract One field per line: "fieldName: description of what to look for"
     * @param modelName      The model to use
     * @param timeoutSeconds Maximum time to wait for processing (default: 60)
     * @return Dictionary where each key is a field name, value is the extracted text
     */
    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input Text", description = "The document, email, invoice, or any text to extract data from")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.TEXTAREA)
        @Pkg(label = "Fields To Extract",
             description = "One field per line in format: fieldName: description\nExample:\n  invoice_number: the invoice ID or reference number\n  total_amount: the total amount due including taxes\n  due_date: the payment due date")
        @NotEmpty
        String fieldsToExtract,

        // Note: gemma3-1b is intentionally NOT offered here (gemma3-270m isn't
        // either). In live testing, gemma3-1b actually extracted the correct
        // values but echoed the literal word "FIELDNAME" from this shared
        // prompt's instructions instead of substituting the real field name
        // (e.g. "FIELDNAME: invoice_number: 12345" instead of
        // "invoice_number: 12345"), which breaks parseExtractionResponse's key
        // matching and makes every field come back NOT_FOUND. Reliable
        // elsewhere (Prompt, ClassifyText, RedactPII, TransformToJSON,
        // SanitizeJSON, SummarizeText) — this is specific to this prompt's
        // literal-placeholder phrasing, which larger models handle fine.
        @Idx(index = "3", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "3.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "3.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "3.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "3.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "3.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "3.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "3.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b"))
        })
        @Pkg(label = "Model", description = "Small Language Model to use for extraction. Qwen2.5 3B recommended for accuracy.", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName,

        @Idx(index = "4", type = AttributeType.NUMBER)
        @Pkg(label = "Timeout (seconds)", description = "Maximum time to wait for processing", default_value = "60", default_value_type = DataType.NUMBER)
        @NotEmpty
        Double timeoutSeconds,

        @Idx(index = "5", type = AttributeType.NUMBER)
        @Pkg(label = "Server Port (Advanced)", description = "Optional: pin the local inference server to a specific TCP port instead of a random one. Only needed in locked-down network/firewall environments that require a fixed, allowlisted port. Leave as 0 for normal automatic behavior.", default_value = "0", default_value_type = DataType.NUMBER)
        Double serverPort

    ) {

        logger.info("ExtractData action started - Model: {}, Fields: {}, Timeout: {}s",
                    modelName, fieldsToExtract.replace("\n", "|"), timeoutSeconds);

        try {
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input text cannot be empty");
            }

            if (fieldsToExtract == null || fieldsToExtract.trim().isEmpty()) {
                throw new BotCommandException("Fields to extract cannot be empty");
            }

            if (timeoutSeconds == null || timeoutSeconds <= 0) {
                timeoutSeconds = 60.0;
            }

            ModelManager.ModelType modelType = ActionUtils.resolveModelType(modelName);

            long startTime = System.currentTimeMillis();
            LlamaInference inference = new LlamaInference(modelType, ActionUtils.resolveServerPort(serverPort));
            long loadTime = System.currentTimeMillis() - startTime;
            if (loadTime > 1000) {
                logger.info("Model loaded in {}ms", loadTime);
            }

            String prompt = buildExtractionPrompt(inputText, fieldsToExtract);
            logger.debug("Generated prompt length: {}", prompt.length());

            int effectiveMaxTokens = Math.min(MAX_OUTPUT_TOKENS, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                EXTRACT_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            LinkedHashMap<String, Value<?>> extracted = parseExtractionResponse(rawResponse, fieldsToExtract);

            long totalTime = System.currentTimeMillis() - startTime;
            int foundCount = (int) extracted.values().stream()
                .filter(v -> !NOT_FOUND.equals(((StringValue) v).get()))
                .count();
            logger.info("Extraction completed in {}ms. {}/{} fields found.",
                totalTime, foundCount, extracted.size());

            return DictionaryHelper.success(extracted, modelType.getId(), totalTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("ExtractData action failed", e);
            String errorMsg;
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                errorMsg = "Extraction timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage() != null && (e.getMessage().contains("model") || e.getMessage().contains("GGUF"))) {
                errorMsg = "Model loading/inference failed: " + e.getMessage() +
                    ". Ensure sufficient memory (~8GB RAM) and model is downloaded.";
            } else {
                errorMsg = "Extraction failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            }
            throw new BotCommandException(errorMsg, e);
        }
    }

    private String buildExtractionPrompt(String inputText, String fieldsToExtract) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Extract the following fields from the text below.\n");
        prompt.append("For each field, output exactly: FIELDNAME: extracted_value\n");
        prompt.append("If a field is not present in the text, output: FIELDNAME: NOT_FOUND\n");
        prompt.append("Output ONLY the field lines, nothing else.\n\n");
        prompt.append("Fields to extract:\n");
        prompt.append(fieldsToExtract.trim()).append("\n\n");
        prompt.append("Text:\n").append(inputText.trim()).append("\n\n");
        prompt.append("Extracted fields:\n");
        return prompt.toString();
    }

    /**
     * Parse the model's field extraction response into a Dictionary.
     * Package-private for unit testing.
     */
    LinkedHashMap<String, Value<?>> parseExtractionResponse(String rawResponse, String fieldsToExtract) {
        // Build ordered map of expected field names (preserving input order)
        LinkedHashMap<String, Value<?>> result = new LinkedHashMap<>();
        for (String line : fieldsToExtract.split("\n")) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String fieldName = line.contains(":") ? line.split(":", 2)[0].trim() : line;
            if (!fieldName.isEmpty()) {
                result.put(fieldName, new StringValue(NOT_FOUND));
            }
        }

        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            logger.warn("Empty extraction response — all fields set to NOT_FOUND");
            return result;
        }

        String cleaned = LlamaInference.stripThinkingBlocks(rawResponse.trim());

        for (String line : cleaned.split("\n")) {
            line = line.trim();
            if (line.isEmpty()) continue;
            if (!line.contains(":")) continue;

            String[] parts = line.split(":", 2);
            String key = parts[0].trim();
            String value = parts.length > 1 ? parts[1].trim() : NOT_FOUND;

            // Case-insensitive match against expected field names
            for (String expectedKey : result.keySet()) {
                if (expectedKey.equalsIgnoreCase(key)) {
                    result.put(expectedKey, new StringValue(value.isEmpty() ? NOT_FOUND : value));
                    break;
                }
            }
        }

        return result;
    }
}
