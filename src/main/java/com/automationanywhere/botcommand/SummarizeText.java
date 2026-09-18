package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.data.Value;
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
 * SummarizeText Action
 *
 * Condenses long text into a clear summary using on-device AI.
 * Perfect for:
 * - Email threads: get the key decision before routing
 * - Support tickets: summarize the issue for tier-2 handoff
 * - Documents: extract the key points without reading everything
 * - Meeting notes: distill action items and decisions
 *
 * Returns a Dictionary with keys:
 * - summary: the condensed text
 * - word_count: word count of the summary
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
    label = "Summarize Text",
    name = "summarizeText",
    description = "Condenses long documents, emails, or tickets into a clear summary using on-device AI",
    node_label = "Summarize: {{inputText}}",
    icon = "pkg.svg",
    comment = true,
    text_color = "#2C3E50",
    background_color = "#FFF8E1",
    allowed_agent_targets = {
        AllowedTarget.WINDOWS,
        AllowedTarget.MAC_OS
    },
    return_type = DICTIONARY,
    return_sub_type = STRING,
    return_required = true
)
public class SummarizeText {

    private static final Logger logger = LogManager.getLogger(SummarizeText.class);

    // Token budgets per summary length. Capped per model via Math.min at call time.
    static final int MAX_OUTPUT_TOKENS_SHORT = 250;
    static final int MAX_OUTPUT_TOKENS_DETAILED = 500;

    // Moderate temperature — some variation is fine for summaries
    static final float SUMMARY_TEMPERATURE = 0.3f;

    @Execute
    public DictionaryValue execute(

        @Idx(index = "1", type = AttributeType.TEXTAREA)
        @Pkg(label = "Input Text", description = "The document, email thread, ticket, or any text to summarize")
        @NotEmpty
        String inputText,

        @Idx(index = "2", type = AttributeType.SELECT, options = {
            @Idx.Option(index = "2.1", pkg = @Pkg(label = "One sentence", value = "one-sentence")),
            @Idx.Option(index = "2.2", pkg = @Pkg(label = "Short (2-3 sentences)", value = "short")),
            @Idx.Option(index = "2.3", pkg = @Pkg(label = "Detailed (full paragraph)", value = "detailed"))
        })
        @Pkg(label = "Summary Length", description = "How much detail to include in the summary", default_value = "short", default_value_type = DataType.STRING)
        @NotEmpty
        String summaryLength,

        @Idx(index = "3", type = AttributeType.TEXT)
        @Pkg(label = "Focus Area", description = "Optional: focus the summary on specific aspects (e.g., 'action items', 'technical issues', 'financial impact')")
        String focusArea,

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
        @Pkg(label = "Model", description = "Small Language Model to use for summarization.", default_value = "qwen3-4b", default_value_type = DataType.STRING)
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

        logger.info("SummarizeText action started - Model: {}, Length: {}, Timeout: {}s",
                    modelName, summaryLength, timeoutSeconds);

        try {
            if (inputText == null || inputText.trim().isEmpty()) {
                throw new BotCommandException("Input text cannot be empty");
            }

            if (summaryLength == null || summaryLength.trim().isEmpty()) {
                summaryLength = "short";
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

            String prompt = buildSummaryPrompt(inputText, summaryLength, focusArea);
            logger.debug("Generated prompt length: {}", prompt.length());

            int tokenBudget = summaryLength.equals("detailed") ? MAX_OUTPUT_TOKENS_DETAILED : MAX_OUTPUT_TOKENS_SHORT;
            int effectiveMaxTokens = Math.min(tokenBudget, modelType.getMaxOutputTokens());
            String rawResponse = inference.generateText(
                prompt,
                effectiveMaxTokens,
                SUMMARY_TEMPERATURE,
                timeoutSeconds.intValue()
            );

            String summary = parseSummaryResponse(rawResponse);

            long totalTime = System.currentTimeMillis() - startTime;
            int wordCount = summary.split("\\s+").length;
            logger.info("Summarization completed in {}ms. Summary: {} words", totalTime, wordCount);

            LinkedHashMap<String, Value<?>> fields = new LinkedHashMap<>();
            fields.put("summary", new StringValue(summary));
            fields.put("word_count", new StringValue(String.valueOf(wordCount)));
            return DictionaryHelper.success(fields, modelType.getId(), totalTime);

        } catch (BotCommandException e) {
            throw e;
        } catch (Exception e) {
            logger.error("SummarizeText action failed", e);
            String errorMsg;
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                errorMsg = "Summarization timed out after " + timeoutSeconds + " seconds. Try increasing timeout or using a smaller model.";
            } else if (e.getMessage() != null && (e.getMessage().contains("model") || e.getMessage().contains("GGUF"))) {
                errorMsg = "Model loading/inference failed: " + e.getMessage() +
                    ". Ensure sufficient memory (~8GB RAM) and model is downloaded.";
            } else {
                errorMsg = "Summarization failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            }
            throw new BotCommandException(errorMsg, e);
        }
    }

    private String buildSummaryPrompt(String inputText, String summaryLength, String focusArea) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Summarize the following text");

        switch (summaryLength.toLowerCase().trim()) {
            case "one-sentence":
                prompt.append(" in exactly ONE sentence");
                break;
            case "detailed":
                prompt.append(" in a detailed paragraph");
                break;
            case "short":
            default:
                prompt.append(" in 2-3 sentences");
                break;
        }

        if (focusArea != null && !focusArea.trim().isEmpty()) {
            prompt.append(", focusing on: ").append(focusArea.trim());
        }

        prompt.append(".\nOutput ONLY the summary, nothing else.\n\n");
        prompt.append("Text:\n").append(inputText.trim()).append("\n\n");
        prompt.append("Summary:");
        return prompt.toString();
    }

    /**
     * Clean the model's summary response.
     * Package-private for unit testing.
     */
    String parseSummaryResponse(String rawResponse) {
        if (rawResponse == null || rawResponse.trim().isEmpty()) {
            throw new BotCommandException(
                "Model returned an empty summary. Try increasing the timeout or using a different model.");
        }

        String cleaned = LlamaInference.stripThinkingBlocks(rawResponse.trim());
        cleaned = cleaned.replaceAll("^(Summary:|Result:|Output:)\\s*", "");
        return cleaned.trim();
    }
}
