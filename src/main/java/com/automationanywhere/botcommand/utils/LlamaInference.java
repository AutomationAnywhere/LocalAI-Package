package com.automationanywhere.botcommand.utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Wrapper around llama-server for text generation.
 *
 * Replaces the de.kherud:llama JNI approach with subprocess-based inference
 * via the official llama.cpp llama-server binary. This supports all current
 * and future models regardless of architecture (qwen3, gemma4, etc.).
 */
public class LlamaInference {
    private static final Logger logger = LogManager.getLogger(LlamaInference.class);

    private final ModelManager.ModelType modelType;
    private final Integer portOverride;

    private static final float TEMPERATURE = 0.1f;
    private static final int MAX_TOKENS = 100;

    public LlamaInference(ModelManager.ModelType modelType) throws Exception {
        this(modelType, null);
    }

    /**
     * @param portOverride optional fixed port for the inference server (see
     *                     LlamaServerManager.ensureModelLoaded); null/&lt;=0
     *                     for the default automatic behavior.
     */
    public LlamaInference(ModelManager.ModelType modelType, Integer portOverride) throws Exception {
        this.modelType = modelType;
        this.portOverride = portOverride;
    }

    /**
     * Starts (or reuses) llama-server for this model with a context window big
     * enough for this request. Deferred from the constructor so the window can
     * be sized to the actual prompt — see LlamaServerManager.MIN_CONTEXT.
     */
    private void ensureLoaded(String prompt, int maxTokens) throws Exception {
        LlamaServerManager.getInstance().ensureModelLoaded(
            modelType, portOverride, LlamaServerManager.estimateTokenCount(prompt) + maxTokens);
    }

    /**
     * Generate text for a prompt with timeout support.
     */
    public String generate(String prompt, int timeoutSeconds) throws Exception {
        logger.debug("generate() prompt length={}, timeout={}s", prompt.length(), timeoutSeconds);

        String[] defaultStop = { "</s>", "<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<turn|>" };

        ensureLoaded(prompt, MAX_TOKENS);
        String result = LlamaServerManager.getInstance().complete(
            prompt, MAX_TOKENS, TEMPERATURE, defaultStop, timeoutSeconds);

        logger.info("Generation complete ({} chars)", result.length());
        return result;
    }

    /**
     * General-purpose text generation with configurable parameters.
     * No grammar constraint — delegates to the grammar-aware overload with null grammar.
     */
    public String generateText(String prompt, int maxTokens, float temperature, int timeoutSeconds)
            throws Exception {
        return generateText(prompt, maxTokens, temperature, timeoutSeconds, null);
    }

    /**
     * General-purpose text generation with an optional GBNF grammar constraint.
     *
     * When {@code grammar} is non-null, llama-server constrains token sampling so the
     * output is guaranteed to match the grammar (e.g. valid JSON).  Pass one of the
     * constants from {@link JsonGrammar} for structured-output use cases.
     *
     * The prompt should already be fully formatted (chat template applied).
     */
    public String generateText(String prompt, int maxTokens, float temperature,
                               int timeoutSeconds, String grammar) throws Exception {
        logger.debug("generateText() maxTokens={}, temperature={}, timeout={}s, grammar={}",
            maxTokens, temperature, timeoutSeconds, grammar != null ? "set" : "none");

        // Apply chat template if not already formatted
        String formattedPrompt = applyChatTemplate(prompt);

        String[] stopSequences = {
            "\nQ:", "\nAnswer:", "\n\nQ:", "\n\nAnswer:", "\nBased on",
            "<|endoftext|>", "<|im_end|>", "</s>", "<turn|>", "<|turn>"
        };

        int effectiveMaxTokens = Math.min(maxTokens, modelType.getMaxOutputTokens());
        ensureLoaded(formattedPrompt, effectiveMaxTokens);

        String result = LlamaServerManager.getInstance().complete(
            formattedPrompt,
            effectiveMaxTokens,
            temperature,
            stopSequences,
            timeoutSeconds,
            grammar);

        logger.info("generateText complete ({} chars)", result.length());
        return result;
    }

    /**
     * Generate with model-specific sanitization fallback (used by SanitizeJSON).
     */
    public String generateSanitizedJSON(String inputText, int timeoutSeconds) throws Exception {
        String prompt = applyChatTemplate(
            "You are a JSON sanitizer. Remove or escape characters that would break JSON: " +
            "quotes, backslashes, newlines, tabs. Preserve meaning. Input: " + inputText +
            ". Output only the sanitized text.");

        // Outside the try: a load failure (e.g. missing model file) must surface,
        // not be masked by the rule-based fallback below.
        ensureLoaded(prompt, MAX_TOKENS);

        try {
            String result = LlamaServerManager.getInstance().complete(
                prompt, MAX_TOKENS, TEMPERATURE, new String[]{"</s>", "<|im_end|>"}, timeoutSeconds);

            String cleaned = result.trim();
            if (!cleaned.isEmpty() && cleaned.length() < inputText.length() * 3) {
                return cleaned;
            }
            logger.warn("Model output suspicious, using fallback sanitization");
            return sanitizeForJSON(inputText);

        } catch (Exception e) {
            logger.warn("Model generation failed, using fallback: {}", e.getMessage());
            return sanitizeForJSON(inputText);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Chat template formatting
    // ──────────────────────────────────────────────────────────────────────────

    private String applyChatTemplate(String prompt) {
        if (prompt.contains("<|im_start|>") || prompt.contains("<|system|>")
            || prompt.contains("<|user|>")  || prompt.contains("<|assistant|>")
            || prompt.contains("<|turn>")   || prompt.contains("<start_of_turn>")
            || prompt.contains("<|end|>")) {
            logger.debug("Prompt already contains chat template markers, using as-is");
            return prompt;
        }

        switch (modelType.getPromptTemplate()) {
            case CHATML:
                return "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                    + "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                    + "<|im_start|>assistant\n";

            case CHATML_QWEN3:
                // /no_think disables Qwen3's chain-of-thought blocks
                return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    + "<|im_start|>user\n" + prompt + " /no_think<|im_end|>\n"
                    + "<|im_start|>assistant\n";

            case GEMMA3:
                return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n"
                    + "<start_of_turn>model\n";

            case GEMMA4:
                return "<|turn>system\nYou are a helpful assistant<turn|>\n"
                    + "<|turn>user\n" + prompt + "<turn|>\n"
                    + "<|turn>model\n";

            case PHI4:
                return "<|system|>\nYou are a helpful assistant<|end|>\n"
                    + "<|user|>\n" + prompt + "<|end|>\n"
                    + "<|assistant|>\n";

            default: // RAW
                return prompt;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Static utilities
    // ──────────────────────────────────────────────────────────────────────────

    public static String sanitizeForJSON(String input) {
        if (input == null) return "";
        return input
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace("\b", "\\b")
            .replace("\f", "\\f");
    }

    public static String stripThinkingBlocks(String text) {
        if (text == null) return text;
        String cleaned = text.replaceAll("(?s)<think>.*?</think>", "").trim();
        if (cleaned.contains("<think>")) {
            cleaned = cleaned.substring(0, cleaned.indexOf("<think>")).trim();
        }
        return cleaned.replace("</think>", "").trim();
    }

    public ModelManager.ModelType getModelType() {
        return modelType;
    }
}
