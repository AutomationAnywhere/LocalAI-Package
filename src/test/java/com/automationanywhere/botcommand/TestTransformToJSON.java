package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Tests for TransformToJSON action
 * Tests multiple models to ensure prompts work across different model sizes
 */
public class TestTransformToJSON {

    private TransformToJSON transformAction;
    private Gson gson = new Gson();

    @BeforeClass
    public void setUp() {
        transformAction = new TransformToJSON();
        System.out.println("=== TransformToJSON Test Suite ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("===================================");
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
            transformAction.execute("", "csv", "compact", "array", "qwen3-4b", 30.0, 0.0);
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
            transformAction.execute("test", "csv", "compact", "array", "invalid-model", 30.0, 0.0);
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
     * Helper method to validate JSON syntax
     */
    private boolean isValidJSON(String json) {
        try {
            gson.fromJson(json, Object.class);
            return true;
        } catch (JsonSyntaxException e) {
            return false;
        }
    }

    /**
     * ===============================================================================
     * FULL INFERENCE TESTS - TinyLlama
     * ===============================================================================
     */

    @Test
    public void testCSVToJSONArray_TinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] CSV to JSON Array - TinyLlama");
        System.out.println("-".repeat(80));

        String csvInput = "Name,Age,City\nJohn,30,NYC\nJane,25,LA";
        System.out.println("Input CSV:\n" + csvInput);
        System.out.println("Model: TinyLlama 1.1B");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = transformAction.execute(
            csvInput,
            "csv",
            "compact",
            "array",
            "qwen3-4b",
            60.0
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();

        assertTrue(isValidJSON(json), "Output should be valid JSON");
        assertTrue(json.trim().startsWith("["), "Should start with [");
        assertTrue(json.trim().endsWith("]"), "Should end with ]");
    }

    @Test
    public void testKeyValueToJSON_TinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Key-Value to JSON - TinyLlama");
        System.out.println("-".repeat(80));

        String kvInput = "Name: John Smith\nAge: 30\nCity: New York";
        System.out.println("Input Key-Value:\n" + kvInput);
        System.out.println("Model: TinyLlama 1.1B");

        DictionaryValue result = transformAction.execute(
            kvInput,
            "key-value",
            "compact",
            "object",
            "qwen3-4b",
            60.0
        , 0.0);

        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();
        assertTrue(isValidJSON(json), "Output should be valid JSON");
    }

    @Test
    public void testListToJSONArray_TinyLlama() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] List to JSON Array - TinyLlama");
        System.out.println("-".repeat(80));

        String listInput = "- Apple\n- Banana\n- Orange";
        System.out.println("Input List:\n" + listInput);
        System.out.println("Model: TinyLlama 1.1B");

        DictionaryValue result = transformAction.execute(
            listInput,
            "list",
            "compact",
            "array",
            "qwen3-4b",
            45.0
        , 0.0);

        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();
        assertTrue(isValidJSON(json), "Output should be valid JSON");
        assertTrue(json.trim().startsWith("["), "Should be a JSON array");
    }

    /**
     * ===============================================================================
     * FULL INFERENCE TESTS - Gemma 2B
     * ===============================================================================
     */

    @Test
    public void testCSVToJSONArray_Gemma2B() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] CSV to JSON Array - Gemma 2B");
        System.out.println("-".repeat(80));

        String csvInput = "Name,Age,City\nJohn,30,NYC\nJane,25,LA";
        System.out.println("Input CSV:\n" + csvInput);
        System.out.println("Model: Gemma 2B");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = transformAction.execute(
            csvInput,
            "csv",
            "compact",
            "array",
            "gemma3-4b",
            60.0
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();

        assertTrue(isValidJSON(json), "Output should be valid JSON");
        assertTrue(json.trim().startsWith("["), "Should start with [");
        assertTrue(json.trim().endsWith("]"), "Should end with ]");

        System.out.println("\n✓ Gemma 2B successfully generated JSON array from CSV");
    }

    @Test
    public void testKeyValueToJSON_Gemma2B() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] Key-Value to JSON - Gemma 2B");
        System.out.println("-".repeat(80));

        String kvInput = "Name: John Smith\nAge: 30\nCity: New York";
        System.out.println("Input Key-Value:\n" + kvInput);
        System.out.println("Model: Gemma 2B");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = transformAction.execute(
            kvInput,
            "key-value",
            "compact",
            "object",
            "gemma3-4b",
            60.0
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();
        assertTrue(isValidJSON(json), "Output should be valid JSON");

        System.out.println("\n✓ Gemma 2B successfully generated JSON object from key-value pairs");
    }

    @Test
    public void testListToJSONArray_Gemma2B() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("[TEST] List to JSON Array - Gemma 2B");
        System.out.println("-".repeat(80));

        String listInput = "- Apple\n- Banana\n- Orange";
        System.out.println("Input List:\n" + listInput);
        System.out.println("Model: Gemma 2B");

        long startTime = System.currentTimeMillis();

        DictionaryValue result = transformAction.execute(
            listInput,
            "list",
            "compact",
            "array",
            "gemma3-4b",
            45.0
        , 0.0);

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("\n✓ Completed in " + elapsed + "ms");
        System.out.println("Output JSON: " + ((StringValue) result.get("json")).get());

        assertNotNull(result);
        String json = ((StringValue) result.get("json")).get();
        assertTrue(isValidJSON(json), "Output should be valid JSON");
        assertTrue(json.trim().startsWith("["), "Should be a JSON array");

        System.out.println("\n✓ Gemma 2B successfully generated JSON array from list");
    }
}
