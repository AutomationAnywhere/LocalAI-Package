package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Tests for Qwen2.5-3B model.
 * Diagnoses timeout and performance issues.
 */
public class TestQwen25 {

    private Prompt promptAction;
    private ValidateDevice validateAction;

    @BeforeClass
    public void setUp() {
        promptAction = new Prompt();
        validateAction = new ValidateDevice();
        System.out.println("=".repeat(80));
        System.out.println("=== Qwen2.5-3B Diagnostic Test Suite ===");
        System.out.println("=".repeat(80));
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("Available Processors: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Max Memory: " + (Runtime.getRuntime().maxMemory() / (1024*1024)) + "MB");
        System.out.println("=".repeat(80));
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test suite completed - models unloaded");
        System.out.println("=".repeat(80));
    }

    @Test(priority = 1)
    public void test1_ValidateQwenModelExists() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST 1] Validate Qwen2.5-3B Model Availability");
        System.out.println("-".repeat(80));

        System.out.println("Checking if qwen3-4b is in ModelType enum...");
        ModelManager.ModelType type = ModelManager.ModelType.fromId("qwen3-4b");
        assertNotNull(type, "Qwen2.5-3B should be in ModelType enum");

        System.out.println("✓ Model ID: " + type.getId());
        System.out.println("✓ File name: " + type.getFileName());
        System.out.println("✓ Size: " + type.getSizeMB() + "MB");
        System.out.println("✓ Context window: " + type.getContextWindow() + " tokens");
        System.out.println("✓ Max output: " + type.getMaxOutputTokens() + " tokens");

        System.out.println("\n✓ Qwen2.5-3B model configuration is valid!");
    }

    @Test(priority = 2)
    public void test2_DownloadQwenModel() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST 2] Download/Validate Qwen2.5-3B Model");
        System.out.println("-".repeat(80));
        System.out.println("This will download ~2.1GB if not already present");
        System.out.println("Download may take 5-15 minutes depending on internet speed");
        System.out.println("-".repeat(80));

        long startTime = System.currentTimeMillis();

        DictionaryValue result = validateAction.execute("qwen3-4b");

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\n✓ Validation completed in " + (elapsed/1000) + " seconds");

        assertNotNull(result, "Validation result should not be null");
        assertTrue("true".equals(((StringValue) result.get("supported")).get()), "Model should be validated successfully");

        System.out.println("✓ Qwen2.5-3B model is ready for use!");
    }

    @Test(priority = 3)
    public void test3_SimpleQuestionQwen() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST 3] Simple Question - Qwen2.5-3B");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is the capital of Florida? A:";
        System.out.println("Prompt: " + prompt);
        System.out.println("Timeout: 120 seconds");
        System.out.println("Temperature: 0.3");
        System.out.println("\nStarting inference...");
        System.out.println("(First load may take 30-90 seconds for model initialization)\n");

        long startTime = System.currentTimeMillis();

        try {
            DictionaryValue result = promptAction.execute(
                prompt,
                "qwen3-4b",
                120.0,  // 2 minute timeout
                0.3
            , 0.0);

            long elapsed = System.currentTimeMillis() - startTime;
            String response = ((StringValue) result.get("response")).get();

            System.out.println("\n✓ Completed in " + (elapsed/1000) + " seconds");
            System.out.println("\nModel response:");
            System.out.println("  \"" + response + "\"");
            System.out.println("\nResponse length: " + response.length() + " characters");

            assertNotNull(result, "Result should not be null");
            assertNotNull(response, "Result value should not be null");
            assertFalse(response.isEmpty(), "Response should not be empty");

            if (elapsed < 60000) {
                System.out.println("\n✓ Good performance (< 60 seconds)");
            } else if (elapsed < 120000) {
                System.out.println("\n⚠ Slow performance (60-120 seconds) - this is concerning");
            }

            System.out.println("\n✓ Qwen2.5-3B inference successful!");

        } catch (Exception e) {
            System.err.println("\n✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            fail("Qwen2.5-3B inference failed: " + e.getMessage());
        }
    }

    @Test(priority = 4)
    public void test4_JSONGenerationQwen() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST 4] JSON Generation - Qwen2.5-3B (Optimized Task)");
        System.out.println("-".repeat(80));

        String prompt = "Convert this to JSON: Name is John, Age is 30, City is NYC\n\nOutput only valid JSON:";
        System.out.println("Prompt: " + prompt);
        System.out.println("\nStarting JSON generation test...");
        System.out.println("(Should be faster since model is loaded)\n");

        long startTime = System.currentTimeMillis();

        try {
            DictionaryValue result = promptAction.execute(
                prompt,
                "qwen3-4b",
                120.0,
                0.2  // Low temperature for structured output
            , 0.0);

            long elapsed = System.currentTimeMillis() - startTime;
            String response = ((StringValue) result.get("response")).get();

            System.out.println("\n✓ Completed in " + (elapsed/1000) + " seconds");
            System.out.println("\nModel response:");
            System.out.println("  " + response);

            assertNotNull(result);
            assertNotNull(response);
            assertFalse(response.isEmpty());

            if (elapsed < 30000) {
                System.out.println("\n✓ Excellent performance (< 30 seconds) - model is cached!");
            } else if (elapsed < 60000) {
                System.out.println("\n✓ Acceptable performance (30-60 seconds)");
            } else {
                System.out.println("\n⚠ WARNING: Slow performance (" + (elapsed/1000) + " seconds)");
            }

            System.out.println("\n✓ Qwen2.5-3B JSON generation successful!");

        } catch (Exception e) {
            System.err.println("\n✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            fail("Qwen2.5-3B JSON generation failed: " + e.getMessage());
        }
    }

    @Test(priority = 5)
    public void test5_VeryShortPromptQwen() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST 5] Very Short Prompt - Qwen2.5-3B (Performance Check)");
        System.out.println("-".repeat(80));

        String prompt = "Say hello.";
        System.out.println("Prompt: " + prompt);
        System.out.println("\nTesting minimal prompt...\n");

        long startTime = System.currentTimeMillis();

        try {
            DictionaryValue result = promptAction.execute(
                prompt,
                "qwen3-4b",
                120.0,
                0.7
            , 0.0);

            long elapsed = System.currentTimeMillis() - startTime;
            String response = ((StringValue) result.get("response")).get();

            System.out.println("\n✓ Completed in " + (elapsed/1000) + " seconds");
            System.out.println("\nModel response:");
            System.out.println("  \"" + response + "\"");

            assertNotNull(result);
            assertNotNull(response);
            assertFalse(response.isEmpty());

            System.out.println("\nPerformance Analysis:");
            System.out.println("  - Elapsed time: " + elapsed + "ms");
            System.out.println("  - Response length: " + response.length() + " chars");
            System.out.println("  - Tokens/sec estimate: ~" + (response.length() * 1000 / elapsed) + " chars/sec");

            System.out.println("\n✓ Short prompt test successful!");

        } catch (Exception e) {
            System.err.println("\n✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            fail("Short prompt test failed: " + e.getMessage());
        }
    }

    @Test(priority = 6)
    public void test6_PerformanceSummary() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("QWEN2.5-3B TEST SUMMARY");
        System.out.println("=".repeat(80));
        System.out.println("\nIf all tests passed:");
        System.out.println("  ✓ Model downloads correctly");
        System.out.println("  ✓ Model loads successfully");
        System.out.println("  ✓ Inference works (even if slow)");
        System.out.println("  ✓ Model is suitable for JSON/structured tasks");
        System.out.println("\nIf tests are slow (>60 seconds):");
        System.out.println("  - Check: RAM available (need 5-6GB)");
        System.out.println("  - Check: CPU usage during inference");
        System.out.println("  - Check: Model file integrity");
        System.out.println("  - Consider: Qwen2.5-3B is larger than TinyLlama/Gemma2B");
        System.out.println("\nRecommendation:");
        System.out.println("  - For speed: Use TinyLlama (669MB) or Gemma 2B (1.7GB)");
        System.out.println("  - For quality + context: Use Qwen2.5 3B (2.1GB, 128K context)");
        System.out.println("  - For best quality: Use Gemma 4 E2B (3.1GB, 128K context)");
        System.out.println("=".repeat(80));
    }
}
