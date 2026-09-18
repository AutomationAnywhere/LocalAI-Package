package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Tests for NormalizeAndStandardize action
 */
public class TestNormalizeAndStandardize {

    private NormalizeAndStandardize normalizeAction;

    @BeforeClass
    public void setUp() {
        normalizeAction = new NormalizeAndStandardize();
        System.out.println("=== NormalizeAndStandardize Test Suite ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("==========================================");
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
            normalizeAction.execute("", "phone", "digits only", "qwen3-4b", false, 30.0, 0.0);
            fail("Should throw exception for empty text");

        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
            assertTrue(e.getMessage().contains("cannot be empty"), "Error should mention empty text");
        }
    }

    /**
     * Test invalid model name
     */
    @Test
    public void testInvalidModelName() {
        System.out.println("\n[TEST] Invalid model name");

        try {
            normalizeAction.execute("Test", "phone", "digits only", "invalid-model", false, 30.0, 0.0);
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
    public void testPhoneNormalization() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Phone Number Normalization");
        System.out.println("-".repeat(80));

        try {
            String input = "(555) 123-4567";
            String format = "digits only";

            System.out.println("Input: " + input);
            System.out.println("Format: " + format);

            long startTime = System.currentTimeMillis();

            DictionaryValue result = normalizeAction.execute(
                input,
                "phone",
                format,
                "qwen3-4b",
                true,
                60.0
            , 0.0);

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("\n✓ Completed in " + elapsed + "ms");
            System.out.println("Output: " + ((StringValue) result.get("result")).get());

            assertNotNull(result);
            assertFalse(((StringValue) result.get("result")).get().isEmpty());

        } catch (Exception e) {
            System.out.println("\n✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Phone normalization failed: " + e.getMessage());
        }
    }

    @Test
    public void testDateNormalization() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Date Normalization");
        System.out.println("-".repeat(80));

        try {
            String input = "March 15, 2024";
            String format = "YYYY-MM-DD";

            System.out.println("Input: " + input);
            System.out.println("Expected format: " + format);

            DictionaryValue result = normalizeAction.execute(
                input,
                "date",
                format,
                "qwen3-4b",
                true,
                45.0
            , 0.0);

            System.out.println("Output: " + ((StringValue) result.get("result")).get());

            assertNotNull(result);
            assertFalse(((StringValue) result.get("result")).get().isEmpty());

        } catch (Exception e) {
            System.out.println("\n✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Date normalization failed: " + e.getMessage());
        }
    }

    @Test
    public void testNameNormalization() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Name Normalization");
        System.out.println("-".repeat(80));

        try {
            String input = "SMITH, JOHN Q.";
            String format = "Proper case";

            System.out.println("Input: " + input);
            System.out.println("Format: " + format);

            DictionaryValue result = normalizeAction.execute(
                input,
                "name",
                format,
                "qwen3-4b",
                true,
                45.0
            , 0.0);

            System.out.println("Output: " + ((StringValue) result.get("result")).get());

            assertNotNull(result);
            assertFalse(((StringValue) result.get("result")).get().isEmpty());

        } catch (Exception e) {
            System.out.println("\n✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            fail("Name normalization failed: " + e.getMessage());
        }
    }
}
