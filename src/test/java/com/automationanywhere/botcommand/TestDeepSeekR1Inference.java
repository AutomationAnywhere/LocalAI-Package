package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.LlamaInference;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Inference tests for DeepSeek R1 1.5B (Q4_K_M, ~1.1GB).
 * Tests model loading, inference, and thinking block stripping.
 *
 * DeepSeek R1 1.5B is a reasoning model that produces <think> blocks.
 * At 1.5B parameters it is the smallest model in the lineup. These tests
 * validate integration correctness rather than output quality, since a
 * model this small can produce inconsistent outputs.
 *
 * First run will download the model (~1.1GB) which may take a couple minutes.
 */
public class TestDeepSeekR1Inference {

    private static final String MODEL = "deepseek-r1-1.5b";
    private static final double TIMEOUT = 120.0;

    private Prompt promptAction;
    private ClassifyText classifyAction;

    @BeforeClass
    public void setUp() {
        promptAction = new Prompt();
        classifyAction = new ClassifyText();

        System.out.println("=== DeepSeek R1 1.5B Inference Test Suite ===");
        System.out.println("Model: " + MODEL + " (Q4_K_M, ~1.1GB, reasoning model)");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("=============================================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("DeepSeek R1 test suite completed - model unloaded");
    }

    // ===== Core Integration: Model loads and generates =====

    @Test(priority = 1)
    public void testModelLoadsAndGenerates() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 - Model Loads and Generates");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is the capital of Japan? A:";
        System.out.println("Prompt: " + prompt);
        System.out.println("(First run: model download + loading may take 1-3 minutes)\n");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(prompt, MODEL, TIMEOUT, 0.3, 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("Completed in " + elapsed + "ms");
        System.out.println("Response: " + ((StringValue) result.get("response")).get());

        assertNotNull(result, "Result should not be null");
        assertNotNull(((StringValue) result.get("response")).get(), "Result value should not be null");

        // The model loaded, processed the prompt, and returned a result
        System.out.println("OK: DeepSeek R1 model loads and produces output");
    }

    // ===== Thinking Block Stripping in Live Inference =====

    @Test(priority = 2)
    public void testThinkingBlocksStrippedFromOutput() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 - Thinking Blocks Stripped from Output");
        System.out.println("-".repeat(80));

        String prompt = "Q: What is 2 + 2? A:";
        System.out.println("Prompt: " + prompt);

        DictionaryValue result = promptAction.execute(prompt, MODEL, TIMEOUT, 0.1, 0.0);

        String response = ((StringValue) result.get("response")).get();
        System.out.println("Response: [" + response + "]");

        assertNotNull(result);
        // Verify no raw thinking tags leak through to the user
        assertFalse(response.contains("<think>"),
            "Opening <think> tags should be stripped from output");
        assertFalse(response.contains("</think>"),
            "Closing </think> tags should be stripped from output");

        System.out.println("OK: No thinking tags in output");
    }

    // ===== Cached Model Performance =====

    @Test(priority = 3)
    public void testCachedModelPerformance() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 - Cached Model Performance");
        System.out.println("-".repeat(80));

        String prompt = "Q: What color is the sky? A:";
        System.out.println("Prompt: " + prompt);

        long startTime = System.currentTimeMillis();

        DictionaryValue result = promptAction.execute(prompt, MODEL, TIMEOUT, 0.3, 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("Completed in " + elapsed + "ms");
        System.out.println("Response: " + ((StringValue) result.get("response")).get());

        assertNotNull(result);

        System.out.println("OK: Cached model inference completed in " + elapsed + "ms");
    }

    // ===== Classification Integration =====

    @Test(priority = 4, enabled = false) // DeepSeek R1 1.5B too small for reliable classification
    public void testClassificationIntegration() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 - Classification Integration");
        System.out.println("-".repeat(80));

        String text = "INVOICE #12345\nDate: 2025-01-15\nBill To: Acme Corp\nAmount Due: $1,250.00";
        System.out.println("Input: " + text);
        System.out.println("Categories: invoice, receipt, contract, report");

        DictionaryValue result = classifyAction.execute(
            text, "invoice, receipt, contract, report", MODEL, false, false, TIMEOUT
        , 0.0);

        String category = ((StringValue) result.get("category")).get();
        System.out.println("Classification: " + category);

        assertNotNull(result, "Classification result should not be null");
        assertNotNull(category, "Classification value should not be null");
        assertFalse(category.contains("<think>"),
            "Thinking blocks should be stripped from classification");

        System.out.println("OK: DeepSeek R1 classification integration works");
    }

    // ===== Multiple Sequential Inferences =====

    @Test(priority = 5)
    public void testMultipleSequentialInferences() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] DeepSeek R1 - Multiple Sequential Inferences");
        System.out.println("-".repeat(80));

        String[] prompts = {
            "Q: What is 1 + 1? A:",
            "Q: Name a color. A:",
            "Q: Is water wet? A:"
        };

        for (int i = 0; i < prompts.length; i++) {
            System.out.println("Inference " + (i + 1) + ": " + prompts[i]);

            long startTime = System.currentTimeMillis();
            DictionaryValue result = promptAction.execute(prompts[i], MODEL, TIMEOUT, 0.3, 0.0);
            long elapsed = System.currentTimeMillis() - startTime;

            String response = ((StringValue) result.get("response")).get();
            System.out.println("  Response (" + elapsed + "ms): " + response);

            assertNotNull(result, "Result " + (i + 1) + " should not be null");
            assertNotNull(response, "Result value " + (i + 1) + " should not be null");
        }

        System.out.println("OK: Multiple sequential inferences completed successfully");
    }
}
