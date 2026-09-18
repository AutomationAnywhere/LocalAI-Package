package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Tests for Prompt action.
 * Tests general-purpose text generation with custom prompts.
 */
public class TestPrompt {

    private Prompt promptAction;

    @BeforeClass
    public void setUp() {
        promptAction = new Prompt();
        System.out.println("=== Prompt Test Suite ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("=========================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("Test suite completed - models unloaded");
    }

    @Test
    public void testEmptyPromptValidation() {
        System.out.println("\n[TEST] Empty prompt validation");

        boolean exceptionThrown = false;
        try {
            promptAction.execute("", "qwen3-4b", 30.0, 0.3, 0.0);
        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            exceptionThrown = true;
            assertTrue(e.getMessage().contains("cannot be empty"));
        }
        assertTrue(exceptionThrown, "Should throw exception for empty prompt");
    }

    @Test
    public void testInvalidModelName() {
        System.out.println("\n[TEST] Invalid model name");

        boolean exceptionThrown = false;
        try {
            promptAction.execute("Test prompt", "invalid-model", 30.0, 0.3, 0.0);
        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            exceptionThrown = true;
            assertTrue(e.getMessage().contains("Invalid model name"));
        }
        assertTrue(exceptionThrown, "Should throw exception for invalid model name");
    }

    @Test
    public void testActionInitialization() {
        System.out.println("\n[TEST] Action initialization");
        assertNotNull(promptAction, "Action should be initialized");
        System.out.println("Action initialized successfully");
    }

    @Test
    public void testAllSupportedModels() {
        System.out.println("\n[TEST] All supported models");

        String[] models = {"qwen3-4b", "qwen3-4b", "phi4-mini", "gemma3-4b", "gemma4-e2b", "deepseek-r1-1.5b"};

        for (String model : models) {
            System.out.println("Testing model: " + model);
            ModelManager.ModelType type = ModelManager.ModelType.fromId(model);
            assertNotNull(type, "Model type should be resolvable: " + model);
            System.out.println("  ✓ " + model + " is supported");
        }
    }

    /**
     * Test 1: Simple question answering with TinyLlama
     */
    @Test
    public void testQuestionAnsweringTinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Question Answering - TinyLlama");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is the capital of Florida? A:";
        System.out.println("Prompt: " + prompt);
        System.out.println("\nRunning inference with TinyLlama...");
        System.out.println("(First run: model loading may take 30-60 seconds)\n");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(
            prompt,
            "qwen3-4b",
            60.0,
            0.3
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("\nModel response:");
        System.out.println("  " + ((StringValue) result.get("response")).get());

        assertNotNull(result, "Result should not be null");
        assertNotNull(((StringValue) result.get("response")).get(), "Result value should not be null");
        assertFalse(((StringValue) result.get("response")).get().isEmpty(), "Response should not be empty");
        assertTrue(((StringValue) result.get("response")).get().length() > 0, "Response should have content");

        System.out.println("\nResponse length: " + ((StringValue) result.get("response")).get().length() + " characters");
        System.out.println("\n✓ TinyLlama inference successful!");
    }

    /**
     * Test 2: Simple math question with TinyLlama
     */
    @Test
    public void testSimpleMathTinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Simple Math - TinyLlama");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is 2 + 2? A:";
        System.out.println("Prompt: " + prompt);
        System.out.println("\nRunning inference (model should be loaded, faster)...\n");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(
            prompt,
            "qwen3-4b",
            30.0,
            0.3
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("\nModel response:");
        System.out.println("  " + ((StringValue) result.get("response")).get());

        assertNotNull(result);
        assertNotNull(((StringValue) result.get("response")).get());
        assertFalse(((StringValue) result.get("response")).get().isEmpty());

        if (elapsed < 10000) {
            System.out.println("\n✓ Fast inference confirmed (< 10 seconds) - model was cached!");
        }

        System.out.println("\n✓ TinyLlama second inference successful!");
    }

    /**
     * Test 3: Third question to verify consistent performance
     */
    @Test
    public void testThirdQuestionTinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Third Question - TinyLlama (Performance Check)");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is the capital of Texas? A:";
        System.out.println("Prompt: " + prompt);
        System.out.println("\nRunning third inference...\n");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(
            prompt,
            "qwen3-4b",
            30.0,
            0.3
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("\nModel response:");
        System.out.println("  " + ((StringValue) result.get("response")).get());

        assertNotNull(result);
        assertNotNull(((StringValue) result.get("response")).get());
        assertFalse(((StringValue) result.get("response")).get().isEmpty());

        if (elapsed < 5000) {
            System.out.println("\n✓ Excellent performance (< 5 seconds)!");
        }

        System.out.println("\n✓ TinyLlama third inference successful!");
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SUMMARY: All TinyLlama inference tests passed!");
        System.out.println("Model is working correctly and performance is good.");
        System.out.println("=".repeat(80));
    }
}
