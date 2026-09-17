package com.automationanywhere.botcommand.utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Manages GGUF model file paths and provides the ModelType registry.
 *
 * Model loading and inference are handled by LlamaServerManager, which runs
 * the official llama.cpp llama-server binary as a subprocess. This avoids the
 * JNI binding (de.kherud:llama) that was pinned to llama.cpp b4916 and did
 * not support qwen3, gemma4, or other architectures added after March 2025.
 *
 * Model storage locations:
 *   Windows: %LOCALAPPDATA%\AutomationAnywhere\LocalAI\
 *   macOS:   ~/localAI/
 */
public class ModelManager {
    private static final Logger logger = LogManager.getLogger(ModelManager.class);
    private static volatile ModelManager instance;

    // Platform-specific model cache directory.
    // Windows uses AppData\Local to avoid Controlled Folder Access (CFA) restrictions
    // that can affect files stored directly under the user home root.
    private static final Path MODEL_CACHE_DIR;
    static {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("windows")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData == null || localAppData.isEmpty()) {
                localAppData = System.getProperty("user.home") + "\\AppData\\Local";
            }
            MODEL_CACHE_DIR = Paths.get(localAppData, "AutomationAnywhere", "LocalAI");
        } else {
            MODEL_CACHE_DIR = Paths.get(System.getProperty("user.home"), "localAI");
        }
    }

    /**
     * All supported GGUF models.
     *
     * To add a new model, add a new enum entry here only — all action dropdowns
     * read from this enum via ActionUtils.resolveModelType().
     */
    public enum ModelType {
        // Gemma 3 4B IT - Q4_K_M (~2.49GB, 128K context)
        GEMMA3_4B(
            "gemma3-4b", "gemma3-4b-q4", "gemma-3-4b-it-Q4_K_M.gguf",
            "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
            2490, 131072, 8192, PromptTemplate.GEMMA3),

        // Qwen3 4B - Q4_K_M (~2.5GB, 32K context, ChatML + /no_think)
        QWEN3_4B(
            "qwen3-4b", "qwen3-4b-q4", "Qwen3-4B-Q4_K_M.gguf",
            "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
            2500, 32768, 8192, PromptTemplate.CHATML_QWEN3),

        // Llama 3.2 3B Instruct - Q4_K_M (~2.0GB, 8K context)
        LLAMA3_2_3B(
            "llama3.2-3b", "llama3.2-3b-q4", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            2020, 8192, 4096, PromptTemplate.RAW),

        // Phi-4 Mini Instruct - Q4_K_M (~2.49GB, 128K context)
        PHI4_MINI(
            "phi4-mini", "phi4-mini-q4", "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
            2490, 131072, 8192, PromptTemplate.PHI4),

        // Gemma 4 E2B Instruct - Q4_K_M (~3.11GB, 128K context)
        GEMMA4_E2B(
            "gemma4-e2b", "gemma4-e2b-q4", "gemma-4-E2B-it-Q4_K_M.gguf",
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
            3185, 131072, 4096, PromptTemplate.GEMMA4),

        // Qwen2.5-Coder 3B Instruct - Q4_K_M (~1.93GB, 32K context)
        QWEN2_5_CODER_3B(
            "qwen2.5-coder-3b", "qwen2.5-coder-3b-q4", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
            1977, 32768, 8192, PromptTemplate.CHATML),

        // DeepSeek R1 Distill Qwen 1.5B - Q4_K_M (~1.12GB, reasoning)
        DEEPSEEK_R1_1_5B(
            "deepseek-r1-1.5b", "deepseek-r1-1.5b-q4", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            1147, 131072, 4096, PromptTemplate.CHATML),

        // Gemma 3 270M IT - Q4_K_M (~253MB, 32K context). Ultra-small, built for
        // narrow single-purpose tasks (classification, short extraction), not
        // general-purpose chat. Uses the same GEMMA3 chat template as gemma3-4b.
        GEMMA3_270M(
            "gemma3-270m", "gemma3-270m-q4", "google_gemma-3-270m-it-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q4_K_M.gguf",
            253, 32768, 4096, PromptTemplate.GEMMA3);

        public enum PromptTemplate {
            RAW, CHATML, CHATML_QWEN3, GEMMA3, GEMMA4, PHI4
        }

        private final String id, dirName, fileName, downloadUrl;
        private final int sizeMB, contextWindow, maxOutputTokens;
        private final PromptTemplate promptTemplate;

        ModelType(String id, String dirName, String fileName, String downloadUrl,
                  int sizeMB, int contextWindow, int maxOutputTokens, PromptTemplate promptTemplate) {
            this.id = id; this.dirName = dirName; this.fileName = fileName;
            this.downloadUrl = downloadUrl; this.sizeMB = sizeMB;
            this.contextWindow = contextWindow; this.maxOutputTokens = maxOutputTokens;
            this.promptTemplate = promptTemplate;
        }

        public String getId() { return id; }
        public String getDirName() { return dirName; }
        public String getFileName() { return fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public int getSizeMB() { return sizeMB; }
        public int getContextWindow() { return contextWindow; }
        public int getMaxOutputTokens() { return maxOutputTokens; }
        public PromptTemplate getPromptTemplate() { return promptTemplate; }

        public static ModelType fromId(String id) {
            for (ModelType t : values()) {
                if (t.id.equalsIgnoreCase(id)) return t;
            }
            throw new IllegalArgumentException("Unknown model type: " + id);
        }

        public static String supportedModelIds() {
            StringBuilder sb = new StringBuilder();
            for (ModelType t : values()) {
                if (sb.length() > 0) sb.append(", ");
                sb.append("'").append(t.id).append("'");
            }
            return sb.toString();
        }
    }

    private ModelManager() {
        logger.info("ModelManager initialized");
        logger.info("Model cache directory: {}", MODEL_CACHE_DIR.toAbsolutePath());
        ensureCacheDirectoryExists();
    }

    public static ModelManager getInstance() {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) instance = new ModelManager();
            }
        }
        return instance;
    }

    /** Resolves the filesystem path for a model's GGUF file. */
    public Path getModelPath(ModelType modelType) {
        return MODEL_CACHE_DIR.resolve(modelType.getDirName()).resolve(modelType.getFileName());
    }

    /** Resolves the directory that contains a model's GGUF file. */
    public Path getModelDirectory(ModelType modelType) {
        return MODEL_CACHE_DIR.resolve(modelType.getDirName());
    }

    public static Path getModelCacheDir() {
        return MODEL_CACHE_DIR;
    }

    /** Delegates to LlamaServerManager to stop the running inference server. */
    public void shutdown() {
        LlamaServerManager.getInstance().shutdown();
    }

    private void ensureCacheDirectoryExists() {
        try {
            if (!Files.exists(MODEL_CACHE_DIR)) {
                Files.createDirectories(MODEL_CACHE_DIR);
                logger.info("Created model cache directory: {}", MODEL_CACHE_DIR);
            }
        } catch (Exception e) {
            logger.error("Failed to create cache directory", e);
            throw new RuntimeException("Failed to create cache directory: " + MODEL_CACHE_DIR, e);
        }
    }
}
