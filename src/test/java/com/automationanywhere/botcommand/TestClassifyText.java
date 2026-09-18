package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Tests for ClassifyText action
 */
public class TestClassifyText {

    private ClassifyText classifyAction;

    @BeforeClass
    public void setUp() {
        classifyAction = new ClassifyText();
        System.out.println("=== ClassifyText Test Suite ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("================================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("Test suite completed - models unloaded");
    }

    /**
     * Test empty text validation
     */
    @Test
    public void testEmptyTextValidation() {
        System.out.println("\n[TEST] Empty text validation");

        try {
            classifyAction.execute("", "urgent, normal, low", "qwen3-4b", false, false, 30.0, 0.0);
            fail("Should throw exception for empty text");

        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            assertTrue(e.getMessage().contains("cannot be empty"), "Error should mention empty text");
        }
    }

    /**
     * Test empty categories validation
     */
    @Test
    public void testEmptyCategoriesValidation() {
        System.out.println("\n[TEST] Empty categories validation");

        try {
            classifyAction.execute("Test message", "", "qwen3-4b", false, false, 30.0, 0.0);
            fail("Should throw exception for empty categories");

        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            assertTrue(e.getMessage().contains("cannot be empty"), "Error should mention empty categories");
        }
    }

    /**
     * Test invalid model name
     */
    @Test
    public void testInvalidModelName() {
        System.out.println("\n[TEST] Invalid model name");

        try {
            classifyAction.execute("Test", "cat1, cat2", "invalid-model", false, false, 30.0, 0.0);
            fail("Should throw exception for invalid model name");

        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            assertTrue(e.getMessage().contains("Invalid model name"), "Error should mention invalid model name");
        }
    }

    /**
     * Test all supported models
     */
    @Test
    public void testAllSupportedModels() {
        System.out.println("\n[TEST] All supported models");

        String[] models = {"qwen3-4b", "qwen3-4b", "phi4-mini", "gemma3-4b"};

        for (String model : models) {
            System.out.println("Testing model: " + model);

            try {
                ModelManager.ModelType type = ModelManager.ModelType.fromId(model);
                assertNotNull(type, "Model type should be resolvable: " + model);
                System.out.println("  ✓ " + model + " is supported");

            } catch (Exception e) {
                fail("Model should be supported: " + model);
            }
        }
    }

    /**
     * ===============================================================================
     * FULL INFERENCE TESTS
     * ===============================================================================
     * These tests require model download
     * ===============================================================================
     */

    @Test
    public void testUrgencyClassification() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Urgency Classification");
        System.out.println("-".repeat(80));

        try {
            String urgentText = "URGENT: Production server is down! Customers cannot access the site.";
            System.out.println("Input: " + urgentText);
            System.out.println("Categories: urgent, normal, low_priority");

            long startTime = System.currentTimeMillis();

            DictionaryValue result = classifyAction.execute(
                urgentText,
                "urgent, normal, low_priority",
                "qwen3-4b",
                false,
                false,
                60.0
            , 0.0);

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("\n✓ Completed in " + elapsed + "ms");
            System.out.println("Classification: " + ((StringValue) result.get("category")).get());

            assertNotNull(result);
            assertFalse(((StringValue) result.get("category")).get().isEmpty());

        } catch (Exception e) {
            System.out.println("\n✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Classification test failed: " + e.getMessage());
        }
    }

    @Test
    public void testSentimentClassification() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Sentiment Classification");
        System.out.println("-".repeat(80));

        try {
            String positiveText = "I absolutely love this product! It works perfectly.";
            System.out.println("Input: " + positiveText);
            System.out.println("Categories: positive, negative, neutral");

            DictionaryValue result = classifyAction.execute(
                positiveText,
                "positive, negative, neutral",
                "qwen3-4b",
                true,  // include confidence
                true,  // include explanation
                45.0
            , 0.0);

            System.out.println("Classification result: " + ((StringValue) result.get("category")).get());

            assertNotNull(result);
            assertFalse(((StringValue) result.get("category")).get().isEmpty());

        } catch (Exception e) {
            System.out.println("\n✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Sentiment classification failed: " + e.getMessage());
        }
    }
}
