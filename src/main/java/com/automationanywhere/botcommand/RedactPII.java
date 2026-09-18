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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.automationanywhere.commandsdk.model.DataType.DICTIONARY;
import static com.automationanywhere.commandsdk.model.DataType.STRING;

/**
 * RedactPII Action
 *
 * Removes or replaces personally identifiable information (PII) from text
 * before it leaves your organization. Uses on-device AI for context-aware
 * redaction that regex alone can't match.
 *
 * Perfect for:
 * - Sanitizing customer emails before logging or archiving
 * - Removing PII before sending text to external APIs
 * - Compliance workflows: GDPR, HIPAA, CCPA pre-processing
 * - Anonymizing support tickets before export
 *
 * PII types supported: names, emails, phone numbers, SSNs, credit cards,
 * addresses, dates of birth, or all of the above.
 *
 * Returns a Dictionary with keys:
 * - redacted_text: the text with PII replaced by the replacement token
 * - items_redacted: estimated count of redacted items
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
    label = "Redact PII",
    name = "redactPII",
    description = "Removes or replaces personally identifiable information from text using on-device AI. Data never leaves your device.",
    node_label = "Redact PII from: {{inputText}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#FCE4EC",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class RedactPII {

    private static final Logger logger = LogManager.getLogger(RedactPII.class);

    static final float REDACT_TEMPERATURE = 0.1f;

    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input Text", description = "The text to redact PII from")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "2.1", pkg = @Pkg(label = "All PII (names, emails, phones, SSNs, cards, addresses, DOBs)", value = "all")),
            @Idx.Option(index = "2.2", pkg = @Pkg(label = "Names only", value = "names")),
            @Idx.Option(index = "2.3", pkg = @Pkg(label = "Email addresses only", value = "emails")),
            @Idx.Option(index = "2.4", pkg = @Pkg(label = "Phone numbers only", value = "phone_numbers")),
            @Idx.Option(index = "2.5", pkg = @Pkg(label = "SSNs only", value = "ssn")),
            @Idx.Option(index = "2.6", pkg = @Pkg(label = "Credit card numbers only", value = "credit_cards")),
            @Idx.Option(index = "2.7", pkg = @Pkg(label = "Physical addresses only", value = "addresses")),
            @Idx.Option(index = "2.8", pkg = @Pkg(label = "Dates of birth only", value = "dates_of_birth"))
        })
        @Pkg(label = "PII Types to Redact", description = "Which types of personal information to remove", default_value = "all", default_value_type = DataType.STRING)
        @NotEmpty
        String piiTypes,

        @Idx(index = "3", type = AttributeType.TEXT)
        @Pkg(label = "Replacement Token", description = "Text to replace PII with (e.g., [REDACTED], ***, <PII>)", default_value = "[REDACTED]", default_value_type = DataType.STRING)
        String replacementToken,

        // Note: gemma3-270m is intentionally NOT offered here. In live testing it
        // returned PII-redaction input completely unmodified (no entities redacted
        // at all) — a real instruction-following failure, not a formatting quirk.
        // It performs well on Prompt and ClassifyText; redaction is a different,
        // harder instruction-following task that a 270M model doesn't handle.
        @Idx(index = "4", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "4.1", pkg = @Pkg(label = "Qwen3 4B (Q4, ~2.5GB, 32K ctx) — best for structured output", value = "qwen3-4b")),
            @Idx.Option(index = "4.2", pkg = @Pkg(label = "Llama 3.2 3B (Q4, ~2.0GB, 8K ctx) — fast, proven baseline", value = "llama3.2-3b")),
            @Idx.Option(index = "4.3", pkg = @Pkg(label = "Phi-4 Mini (Q4, ~2.5GB, 128K ctx) — best for instructions & reasoning", value = "phi4-mini")),
            @Idx.Option(index = "4.4", pkg = @Pkg(label = "Gemma 3 4B (Q4, ~2.5GB, 128K ctx) — balanced all-rounder", value = "gemma3-4b")),
            @Idx.Option(index = "4.5", pkg = @Pkg(label = "Gemma 4 E2B (Q4, ~3.1GB, 128K ctx) — highest quality", value = "gemma4-e2b")),
            @Idx.Option(index = "4.6", pkg = @Pkg(label = "DeepSeek R1 1.5B (Q4, ~1.1GB, 128K ctx) — fastest, chain-of-thought", value = "deepseek-r1-1.5b")),
            @Idx.Option(index = "4.7", pkg = @Pkg(label = "Qwen2.5-Coder 3B (Q4, ~1.9GB, 32K ctx) — code & scripting specialist", value = "qwen2.5-coder-3b")),
            @Idx.Option(index = "4.8", pkg = @Pkg(label = "Gemma 3 1B (Q4, ~769MB, 32K ctx) — small & fast, reliable across tasks", value = "gemma3-1b"))
        })
        @Pkg(label = "Model", description = "Small Language Model to use for redaction. Qwen2.5 3B recommended.", default_value = "qwen3-4b", default_value_type = DataType.STRING)
        @NotEmpty
        String modelName,

        @Idx(index = "5", type = AttributeType.NUMBER)
        @Pkg(label = "Timeout (seconds)", description = "Maximum time to wait for processing", default_value = "45", default_value_type = DataType.NUMBER)
        @NotEmpty
        Double timeoutSeconds,

        @Idx(index = "6", type = AttributeType.NUMBER)
        @Pkg(label = "Server Port (Advanced)", description = "Optional: pin the local inference server to a specific TCP port instead of a random one. Only needed in locked-down network/firewall environments that require a fixed, allowlisted port. Leave as 0 for normal automatic behavior.", default_value = "0", default_value_type = DataType.NUMBER)
        Double serverPort

    ) {

        logger.info("RedactPII action started - Model: {}, PII types: {}, Timeout: {}s",
                    modelName, piiTypes, timeoutSeconds);

        try {
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input text cannot be empty");
            }

            if (piiTypes == null || piiTypes.trim().isEmpty()) {
                piiTypes = "all";
            }

            if (replacementToken == null || replacementToken.isEmpty()) {
                replacementToken = "[REDACTED]";
            }

            if (timeoutSeconds == null || timeoutSeconds <= 0) {
                timeoutSeconds = 45.0;
            }

            ModelManager.ModelType modelType = ActionUtils.resolveModelType(modelName);

            long startTime = System.currentTimeMillis();
            LlamaInference inference = new LlamaInference(modelType, ActionUtils.resolveServerPort(serverPort));
            long loadTime = System.currentTimeMillis() - startTime;
            if (loadTime > 1000) {
                logger.info("Model loaded in {}ms", loadTime);
            }

            String prompt = buildRedactionPrompt(inputText, piiTypes, replacementToken);
            logger.debug("Generated prompt length: {}", prompt.length());

            // Output should be roughly same length as input — cap generously
            int tokenBudget = Math.max(200, inputText.split("\\s+").length * 2);
            int effectiveMaxTokens = Math.min(tokenBudget, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                REDACT_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            String redactedText = parseRedactionResponse(rawResponse, inputText, replacementToken);

            long totalTime = System.currentTimeMillis() - startTime;
            int itemsRedacted = countRedactions(redactedText, replacementToken);
            logger.info("Redaction completed in {}ms. Items redacted: {}", totalTime, itemsRedacted);

            LinkedHashMap<String, Value<?>> fields = new LinkedHashMap<>();
            fields.put("redacted_text", new StringValue(redactedText));
            fields.put("items_redacted", new StringValue(String.valueOf(itemsRedacted)));
            return DictionaryHelper.success(fields, modelType.getId(), totalTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("RedactPII action failed", e);
            String errorMsg;
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                errorMsg = "Redaction timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage() != null && (e.getMessage().contains("model") || e.getMessage().contains("GGUF"))) {
                errorMsg = "Model loading/inference failed: " + e.getMessage() +
                    ". Ensure sufficient memory (~8GB RAM) and model is downloaded.";
            } else {
                errorMsg = "Redaction failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            }
            throw new BotCommandException(errorMsg, e);
        }
    }

    private String buildRedactionPrompt(String inputText, String piiTypes, String replacementToken) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Redact the following types of personal information from the text below.\n");
        prompt.append("Replace each piece of PII with: ").append(replacementToken).append("\n\n");

        prompt.append("PII types to redact:\n");
        switch (piiTypes.toLowerCase().trim()) {
            case "names":
                prompt.append("- Full names and person names\n");
                break;
            case "emails":
                prompt.append("- Email addresses\n");
                break;
            case "phone_numbers":
                prompt.append("- Phone numbers (any format)\n");
                break;
            case "ssn":
                prompt.append("- Social Security Numbers (SSNs)\n");
                break;
            case "credit_cards":
                prompt.append("- Credit card numbers\n");
                break;
            case "addresses":
                prompt.append("- Physical street addresses\n");
                break;
            case "dates_of_birth":
                prompt.append("- Dates of birth\n");
                break;
            case "all":
            default:
                prompt.append("- Full names and person names\n");
                prompt.append("- Email addresses\n");
                prompt.append("- Phone numbers (any format)\n");
                prompt.append("- Social Security Numbers (SSNs)\n");
                prompt.append("- Credit card numbers\n");
                prompt.append("- Physical street addresses\n");
                prompt.append("- Dates of birth\n");
                break;
        }

        prompt.append("\nOutput ONLY the redacted text, preserving all other content exactly.\n\n");
        prompt.append("Text:\n").append(inputText.trim()).append("\n\n");
        prompt.append("Redacted text:");
        return prompt.toString();
    }

    /**
     * Clean the model's redaction response.
     * Package-private for unit testing.
     */
    String parseRedactionResponse(String rawResponse, String originalText, String replacementToken) {
        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            logger.warn("Empty redaction response — returning original text unmodified");
            return originalText;
        }

        String cleaned = LlamaInference.stripThinkingBlocks(rawResponse.trim());
        cleaned = cleaned.replaceAll("^(Redacted text:|Output:|Result:)\\s*", "");
        return cleaned.trim();
    }

    private int countRedactions(String text, String token) {
        if (text == null || text.isEmpty() || token == null || token.isEmpty()) return 0;
        int count = 0;
        int idx = 0;
        while ((idx = text.indexOf(token, idx)) != -1) {
            count++;
            idx += token.length();
        }
        return count;
    }
}
