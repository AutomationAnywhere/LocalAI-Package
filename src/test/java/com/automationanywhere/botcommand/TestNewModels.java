package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.LlamaInference;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.nio.file.Path;

import static org.testng.Assert.*;

/**
 * Tests for Gemma 4 E2B and DeepSeek R1 1.5B model support.
 * Validates model configuration, template formatting, thinking block stripping,
 * and inference capabilities.
 */
public class TestNewModels {

    private Prompt promptAction;

    @BeforeClass
    public void setUp() {
        promptAction = new Prompt();
        System.out.println("=== New Models Test Suite (Gemma 4 E2B + DeepSeek R1 1.5B) ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("==============================================================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("Test suite completed - models unloaded");
    }

    // ===== Model Configuration Tests =====

    @Test
    public void testGemma4E2BModelTypeResolvable() {
        System.out.println("\n[TEST] Gemma 4 E2B model type resolves from ID");
        ModelManager.ModelType type = ModelManager.ModelType.fromId("gemma4-e2b");
        assertNotNull(type, "Should resolve gemma4-e2b");
        assertEquals(type, ModelManager.ModelType.GEMMA4_E2B);
        System.out.println("  OK: gemma4-e2b resolves to GEMMA4_E2B");
    }

    @Test
    public void testDeepSeekR1ModelTypeResolvable() {
        System.out.println("\n[TEST] DeepSeek R1 1.5B model type resolves from ID");
        ModelManager.ModelType type = ModelManager.ModelType.fromId("deepseek-r1-1.5b");
        assertNotNull(type, "Should resolve deepseek-r1-1.5b");
        assertEquals(type, ModelManager.ModelType.DEEPSEEK_R1_1_5B);
        System.out.println("  OK: deepseek-r1-1.5b resolves to DEEPSEEK_R1_1_5B");
    }

    @Test
    public void testGemma4E2BModelConfig() {
        System.out.println("\n[TEST] Gemma 4 E2B model configuration");
        ModelManager.ModelType type = ModelManager.ModelType.GEMMA4_E2B;

        assertEquals(type.getId(), "gemma4-e2b");
        assertEquals(type.getDirName(), "gemma4-e2b-q4");
        assertEquals(type.getFileName(), "gemma-4-E2B-it-Q4_K_M.gguf");
        assertTrue(type.getDownloadUrl().contains("huggingface.co"));
        assertTrue(type.getDownloadUrl().contains("gemma-4-E2B-it-Q4_K_M.gguf"));
        assertEquals(type.getSizeMB(), 3185);
        assertEquals(type.getContextWindow(), 131072);
        assertEquals(type.getMaxOutputTokens(), 4096);
        assertEquals(type.getPromptTemplate(), ModelManager.ModelType.PromptTemplate.GEMMA4);

        System.out.println("  ID: " + type.getId());
        System.out.println("  Dir: " + type.getDirName());
        System.out.println("  File: " + type.getFileName());
        System.out.println("  Size: " + type.getSizeMB() + "MB (~3.1GB)");
        System.out.println("  Context: " + type.getContextWindow());
        System.out.println("  Template: " + type.getPromptTemplate());
        System.out.println("  OK: all config values correct");
    }

    @Test
    public void testDeepSeekR1ModelConfig() {
        System.out.println("\n[TEST] DeepSeek R1 1.5B model configuration");
        ModelManager.ModelType type = ModelManager.ModelType.DEEPSEEK_R1_1_5B;

        assertEquals(type.getId(), "deepseek-r1-1.5b");
        assertEquals(type.getDirName(), "deepseek-r1-1.5b-q4");
        assertEquals(type.getFileName(), "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf");
        assertTrue(type.getDownloadUrl().contains("huggingface.co"));
        assertTrue(type.getDownloadUrl().contains("DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"));
        assertEquals(type.getSizeMB(), 1147);
        assertEquals(type.getContextWindow(), 131072);
        assertEquals(type.getMaxOutputTokens(), 4096);
        assertEquals(type.getPromptTemplate(), ModelManager.ModelType.PromptTemplate.CHATML);

        System.out.println("  ID: " + type.getId());
        System.out.println("  Dir: " + type.getDirName());
        System.out.println("  File: " + type.getFileName());
        System.out.println("  Size: " + type.getSizeMB() + "MB (~1.1GB)");
        System.out.println("  Context: " + type.getContextWindow());
        System.out.println("  Template: " + type.getPromptTemplate());
        System.out.println("  OK: all config values correct");
    }

    @Test
    public void testModelSizesUnder4GB() {
        System.out.println("\n[TEST] New models are under 4GB");
        int maxSizeMB = 4 * 1024;

        ModelManager.ModelType gemma4 = ModelManager.ModelType.GEMMA4_E2B;
        ModelManager.ModelType deepseek = ModelManager.ModelType.DEEPSEEK_R1_1_5B;

        assertTrue(gemma4.getSizeMB() < maxSizeMB,
            "Gemma 4 E2B (" + gemma4.getSizeMB() + "MB) should be under 4GB");
        assertTrue(deepseek.getSizeMB() < maxSizeMB,
            "DeepSeek R1 1.5B (" + deepseek.getSizeMB() + "MB) should be under 4GB");

        System.out.println("  Gemma 4 E2B: " + gemma4.getSizeMB() + "MB < " + maxSizeMB + "MB OK");
        System.out.println("  DeepSeek R1 1.5B: " + deepseek.getSizeMB() + "MB < " + maxSizeMB + "MB OK");
    }

    @Test
    public void testModelPaths() {
        System.out.println("\n[TEST] Model paths are valid");
        ModelManager manager = ModelManager.getInstance();

        Path gemma4Path = manager.getModelPath(ModelManager.ModelType.GEMMA4_E2B);
        Path deepseekPath = manager.getModelPath(ModelManager.ModelType.DEEPSEEK_R1_1_5B);

        assertNotNull(gemma4Path);
        assertNotNull(deepseekPath);
        assertTrue(gemma4Path.isAbsolute());
        assertTrue(deepseekPath.isAbsolute());
        assertTrue(gemma4Path.toString().contains("gemma4-e2b-q4"));
        assertTrue(deepseekPath.toString().contains("deepseek-r1-1.5b-q4"));

        System.out.println("  Gemma 4 path: " + gemma4Path);
        System.out.println("  DeepSeek path: " + deepseekPath);
        System.out.println("  OK: paths are valid and absolute");
    }

    @Test
    public void testSupportedModelIdsIncludesNewModels() {
        System.out.println("\n[TEST] supportedModelIds() includes expected models");
        String supported = ModelManager.ModelType.supportedModelIds();

        assertTrue(supported.contains("gemma4-e2b"), "Should list gemma4-e2b");
        assertTrue(supported.contains("deepseek-r1-1.5b"), "Should list deepseek-r1-1.5b");

        System.out.println("  Supported models: " + supported);
        System.out.println("  OK: expected models listed");
    }

    @Test
    public void testAllModelTypesCount() {
        System.out.println("\n[TEST] Total model count is 8");
        assertEquals(ModelManager.ModelType.values().length, 8,
            "Should have 8 model types (7 original + Gemma 3 270M)");
        System.out.println("  OK: 8 models total");
    }

    // ===== Thinking Block Stripping Tests =====

    @Test
    public void testStripThinkingBlocks_Complete() {
        System.out.println("\n[TEST] stripThinkingBlocks - complete think block");
        String input = "<think>Let me reason about this. The capital of France is Paris.</think>Paris";
        String result = LlamaInference.stripThinkingBlocks(input);
        assertEquals(result, "Paris");
        System.out.println("  Input:  " + input);
        System.out.println("  Output: " + result);
        System.out.println("  OK");
    }

    @Test
    public void testStripThinkingBlocks_Multiline() {
        System.out.println("\n[TEST] stripThinkingBlocks - multiline think block");
        String input = "<think>\nLet me think step by step.\n1. Consider the question.\n2. The answer is 4.\n</think>\n4";
        String result = LlamaInference.stripThinkingBlocks(input);
        assertEquals(result, "4");
        System.out.println("  OK: multiline thinking stripped correctly");
    }

    @Test
    public void testStripThinkingBlocks_Unclosed() {
        System.out.println("\n[TEST] stripThinkingBlocks - unclosed think block");
        String input = "<think>Still thinking about this when generation was cut off";
        String result = LlamaInference.stripThinkingBlocks(input);
        assertEquals(result, "");
        System.out.println("  OK: unclosed thinking block handled");
    }

    @Test
    public void testStripThinkingBlocks_NoThinking() {
        System.out.println("\n[TEST] stripThinkingBlocks - no thinking block");
        String input = "Just a normal response without thinking";
        String result = LlamaInference.stripThinkingBlocks(input);
        assertEquals(result, input);
        System.out.println("  OK: non-thinking response unchanged");
    }

    @Test
    public void testStripThinkingBlocks_Null() {
        System.out.println("\n[TEST] stripThinkingBlocks - null input");
        assertNull(LlamaInference.stripThinkingBlocks(null));
        System.out.println("  OK: null returns null");
    }

    @Test
    public void testStripThinkingBlocks_EmptyThink() {
        System.out.println("\n[TEST] stripThinkingBlocks - empty think block");
        String input = "<think></think>The answer is 42";
        String result = LlamaInference.stripThinkingBlocks(input);
        assertEquals(result, "The answer is 42");
        System.out.println("  OK: empty thinking block stripped");
    }

    // ===== Prompt Template Tests =====

    @Test
    public void testGemma4PromptTemplate() {
        System.out.println("\n[TEST] Gemma 4 uses GEMMA4 prompt template");
        assertEquals(ModelManager.ModelType.GEMMA4_E2B.getPromptTemplate(),
            ModelManager.ModelType.PromptTemplate.GEMMA4);
        System.out.println("  OK: GEMMA4 template assigned");
    }

    @Test
    public void testDeepSeekPromptTemplate() {
        System.out.println("\n[TEST] DeepSeek R1 uses CHATML prompt template");
        assertEquals(ModelManager.ModelType.DEEPSEEK_R1_1_5B.getPromptTemplate(),
            ModelManager.ModelType.PromptTemplate.CHATML);
        System.out.println("  OK: CHATML template assigned");
    }

    // ===== Case-Insensitive Model ID Lookup =====

    @Test
    public void testCaseInsensitiveLookup() {
        System.out.println("\n[TEST] Case-insensitive model ID lookup");

        assertEquals(ModelManager.ModelType.fromId("GEMMA4-E2B"), ModelManager.ModelType.GEMMA4_E2B);
        assertEquals(ModelManager.ModelType.fromId("Gemma4-E2B"), ModelManager.ModelType.GEMMA4_E2B);
        assertEquals(ModelManager.ModelType.fromId("DEEPSEEK-R1-1.5B"), ModelManager.ModelType.DEEPSEEK_R1_1_5B);
        assertEquals(ModelManager.ModelType.fromId("DeepSeek-R1-1.5b"), ModelManager.ModelType.DEEPSEEK_R1_1_5B);

        System.out.println("  OK: case-insensitive lookup works for both models");
    }

    // ===== Invalid Model Rejection =====

    @Test
    public void testInvalidModelNameRejected() {
        System.out.println("\n[TEST] Invalid model name rejected");

        boolean exceptionThrown = false;
        try {
            promptAction.execute("Test", "gemma4-e3b", 30.0, 0.3);
        } catch (Exception e) {
            exceptionThrown = true;
            assertTrue(e.getMessage().contains("Invalid model name"));
        }
        assertTrue(exceptionThrown, "Should reject invalid model name");
        System.out.println("  OK: invalid model name correctly rejected");
    }

    // ===== Inference Tests (disabled by default - require model downloads) =====

    /**
     * Test Gemma 4 E2B inference with a simple question.
     * Enable to test actual inference (requires ~3.1GB download).
     */
    @Test(enabled = false)
    public void testGemma4E2BInference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Gemma 4 E2B Inference");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is the capital of France? A:";
        System.out.println("Prompt: " + prompt);

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(prompt, "gemma4-e2b", 120.0, 0.3);

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\nCompleted in " + elapsed + "ms");
        System.out.println("Response: " + ((StringValue) result.get("response")).get());

        assertNotNull(result);
        assertNotNull(((StringValue) result.get("response")).get());
        assertFalse(((StringValue) result.get("response")).get().isEmpty());
        System.out.println("\nOK: Gemma 4 E2B inference successful");
    }

    /**
     * Test DeepSeek R1 1.5B inference with a simple question.
     * Enable to test actual inference (requires ~1.1GB download).
     */
    @Test(enabled = false)
    public void testDeepSeekR1Inference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 1.5B Inference");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is 15 + 27? A:";
        System.out.println("Prompt: " + prompt);

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(prompt, "deepseek-r1-1.5b", 120.0, 0.3);

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\nCompleted in " + elapsed + "ms");
        System.out.println("Response: " + ((StringValue) result.get("response")).get());

        assertNotNull(result);
        assertNotNull(((StringValue) result.get("response")).get());
        assertFalse(((StringValue) result.get("response")).get().isEmpty());
        System.out.println("\nOK: DeepSeek R1 1.5B inference successful");
    }
}
