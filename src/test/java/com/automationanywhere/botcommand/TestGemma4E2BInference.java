package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.automationanywhere.botcommand.utils.ModelDownloader;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.testng.Assert.*;

/**
 * Tests for Gemma 4 E2B (Q4_K_M, ~3.1GB).
 *
 * Gemma 4 was released April 2, 2026. The gemma4 architecture is supported
 * via a custom java-llama.cpp fork built against llama.cpp b8648.
 *
 * These tests validate:
 * - Model configuration correctness
 * - Model download from HuggingFace
 * - GGUF file integrity
 * - Full inference (question answering, math, classification, JSON, dates)
 */
public class TestGemma4E2BInference {

    private static final String MODEL = "gemma4-e2b";

    private Prompt promptAction;
    private ClassifyText classifyAction;
    private TransformToJSON transformAction;
    private NormalizeAndStandardize normalizeAction;
    private Gson gson = new Gson();

    @BeforeClass
    public void setUp() {
        promptAction = new Prompt();
        classifyAction = new ClassifyText();
        transformAction = new TransformToJSON();
        normalizeAction = new NormalizeAndStandardize();

        System.out.println("=== Gemma 4 E2B Test Suite ===");
        System.out.println("Model: " + MODEL + " (Q4_K_M, ~3.1GB, 5.1B params)");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("=========================================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("Gemma 4 E2B test suite completed - model unloaded");
    }

    // ===== Model Download & File Integrity =====

    @Test(priority = 1)
    public void testModelDownloadAndFileIntegrity() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Gemma 4 E2B - Model Download & File Integrity");
        System.out.println("-".repeat(80));

        ModelManager.ModelType modelType = ModelManager.ModelType.GEMMA4_E2B;
        ModelManager manager = ModelManager.getInstance();
        Path modelPath = manager.getModelPath(modelType);

        System.out.println("Model path: " + modelPath);

        // Download if not present
        if (!Files.exists(modelPath)) {
            System.out.println("Model not found locally, downloading (~3.1GB)...");
            System.out.println("This may take several minutes on first run.");
            try {
                ModelDownloader.downloadModel(modelType);
            } catch (Exception e) {
                fail("Model download failed: " + e.getMessage());
            }
        }

        assertTrue(Files.exists(modelPath), "Model file should exist after download");

        try {
            long fileSize = Files.size(modelPath);
            long expectedMinBytes = (long) (modelType.getSizeMB() * 0.5) * 1024 * 1024; // 50% of expected
            long expectedMaxBytes = (long) (modelType.getSizeMB() * 1.2) * 1024 * 1024; // 120% of expected

            System.out.println("File size: " + (fileSize / 1024 / 1024) + "MB");
            System.out.println("Expected range: " + (expectedMinBytes / 1024 / 1024) + "MB - " + (expectedMaxBytes / 1024 / 1024) + "MB");

            assertTrue(fileSize > expectedMinBytes,
                "File size (" + fileSize + ") should be at least 50% of expected (" + expectedMinBytes + ")");
            assertTrue(fileSize < expectedMaxBytes,
                "File size (" + fileSize + ") should be within 120% of expected (" + expectedMaxBytes + ")");

            System.out.println("OK: Model file exists with correct size");

        } catch (Exception e) {
            fail("Could not check file size: " + e.getMessage());
        }
    }

    @Test(priority = 2)
    public void testModelFileIsValidGGUF() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Gemma 4 E2B - GGUF File Validation");
        System.out.println("-".repeat(80));

        ModelManager.ModelType modelType = ModelManager.ModelType.GEMMA4_E2B;
        Path modelPath = ModelManager.getInstance().getModelPath(modelType);

        if (!Files.exists(modelPath)) {
            System.out.println("Model not downloaded, skipping GGUF validation");
            return;
        }

        try {
            // GGUF files start with magic bytes "GGUF" (0x46475547)
            byte[] header = Files.newInputStream(modelPath).readNBytes(4);
            String magic = new String(header);
            assertEquals(magic, "GGUF", "File should have GGUF magic header");
            System.out.println("GGUF magic header: " + magic + " OK");

        } catch (Exception e) {
            fail("Could not read GGUF header: " + e.getMessage());
        }
    }

    @Test(priority = 3)
    public void testGemma4ArchitectureDetection() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Gemma 4 E2B - Architecture Support Check");
        System.out.println("-".repeat(80));

        // Load the model via llama-server
        try {
            ModelManager.ModelType modelType = ModelManager.ModelType.GEMMA4_E2B;
            new com.automationanywhere.botcommand.utils.LlamaInference(modelType);

            System.out.println("OK: Gemma 4 architecture is supported via llama-server subprocess");

        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
            fail("Gemma 4 model loading failed: " + e.getMessage());
        }
    }

    // ===== Inference Tests =====

    @Test(priority = 10)
    public void testQuestionAnswering() {
        System.out.println("\n[TEST] Gemma 4 E2B - Question Answering");

        String prompt = "Q: What is the capital of France? A:";
        DictionaryValue result = promptAction.execute(prompt, MODEL, 120.0, 0.3, 0.0);

        assertNotNull(result);
        assertNotNull(((StringValue) result.get("response")).get());
        assertFalse(((StringValue) result.get("response")).get().isEmpty());
        System.out.println("Response: " + ((StringValue) result.get("response")).get());
        System.out.println("OK: Gemma 4 E2B question answering works");
    }

    @Test(priority = 10)
    public void testMathReasoning() {
        System.out.println("\n[TEST] Gemma 4 E2B - Math Reasoning");

        DictionaryValue result = promptAction.execute("Q: What is 15 + 27? A:", MODEL, 120.0, 0.1, 0.0);

        assertNotNull(result);
        assertFalse(((StringValue) result.get("response")).get().isEmpty());
        assertTrue(((StringValue) result.get("response")).get().contains("42"), "Should contain correct answer 42");
        System.out.println("Response: " + ((StringValue) result.get("response")).get());
        System.out.println("OK");
    }

    @Test(priority = 10)
    public void testSentimentClassification() {
        System.out.println("\n[TEST] Gemma 4 E2B - Sentiment Classification");

        DictionaryValue result = classifyAction.execute(
            "This product exceeded all my expectations. Absolutely fantastic quality!",
            "positive, negative, neutral", MODEL, true, false, 120.0
        , 0.0);

        assertNotNull(result);
        assertFalse(((StringValue) result.get("category")).get().isEmpty());
        System.out.println("Classification: " + ((StringValue) result.get("category")).get());
        System.out.println("OK");
    }

    @Test(priority = 10)
    public void testCSVToJSON() {
        System.out.println("\n[TEST] Gemma 4 E2B - CSV to JSON");

        DictionaryValue result = transformAction.execute(
            "Name,Age,City\nAlice,28,Boston\nBob,35,Chicago",
            "csv", "compact", "array", MODEL, 120.0
        , 0.0);

        assertNotNull(result);
        assertTrue(isValidJSON(((StringValue) result.get("json")).get()), "Output should be valid JSON");
        System.out.println("JSON: " + ((StringValue) result.get("json")).get());
        System.out.println("OK");
    }

    @Test(priority = 10)
    public void testDateNormalization() {
        System.out.println("\n[TEST] Gemma 4 E2B - Date Normalization");

        DictionaryValue result = normalizeAction.execute(
            "January 15, 2025", "date", "YYYY-MM-DD", MODEL, true, 120.0
        , 0.0);

        assertNotNull(result);
        assertFalse(((StringValue) result.get("result")).get().isEmpty());
        System.out.println("Output: " + ((StringValue) result.get("result")).get());
        System.out.println("OK");
    }

    private boolean isValidJSON(String json) {
        try {
            gson.fromJson(json, Object.class);
            return true;
        } catch (JsonSyntaxException e) {
            return false;
        }
    }
}
