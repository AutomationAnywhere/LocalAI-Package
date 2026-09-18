package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.exception.BotCommandException;
import com.automationanywhere.botcommand.utils.LlamaInference;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.testng.Assert.*;

/**
 * Tests for SanitizeJSON action.
 *
 * Most tests cover the rule-based utility (LlamaInference.sanitizeForJSON)
 * and the parseAndValidateJSON parse/format logic — both run without any model.
 *
 * Model inference tests are marked enabled = false and require a downloaded model.
 */
public class TestSanitizeJSON {

    private SanitizeJSON sanitizeAction;

    @BeforeClass
    public void setUp() {
        sanitizeAction = new SanitizeJSON();
        System.out.println("=== SanitizeJSON Test Suite ===");
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("================================");
    }

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
        System.out.println("SanitizeJSON test suite completed - models unloaded");
    }

    // ===== LlamaInference.sanitizeForJSON utility (no model needed) =====

    @Test
    public void testFallbackSanitizationEscapesQuotes() {
        System.out.println("\n[TEST] sanitizeForJSON escapes double quotes");
        String input    = "He said \"hello\"";
        String result   = LlamaInference.sanitizeForJSON(input);
        System.out.println("Input:  " + input);
        System.out.println("Output: " + result);
        assertTrue(result.contains("\\\""), "Double quotes should be escaped");
    }

    @Test
    public void testFallbackSanitizationEscapesBackslashes() {
        System.out.println("\n[TEST] sanitizeForJSON escapes backslashes");
        String input  = "C:\\Users\\admin\\file.txt";
        String result = LlamaInference.sanitizeForJSON(input);
        System.out.println("Input:  " + input);
        System.out.println("Output: " + result);
        assertTrue(result.contains("\\\\"), "Backslashes should be escaped");
    }

    @Test
    public void testFallbackSanitizationEscapesNewlines() {
        System.out.println("\n[TEST] sanitizeForJSON escapes newlines and tabs");
        String input  = "Line1\nLine2\tTabbed";
        String result = LlamaInference.sanitizeForJSON(input);
        System.out.println("Input (repr): " + input.replace("\n", "\\n").replace("\t", "\\t"));
        System.out.println("Output: " + result);
        assertFalse(result.contains("\n"), "Newlines should be escaped");
        assertFalse(result.contains("\t"), "Tabs should be escaped");
    }

    @Test
    public void testFallbackSanitizationNullAndEmpty() {
        System.out.println("\n[TEST] sanitizeForJSON handles null and empty");
        assertEquals(LlamaInference.sanitizeForJSON(null), "", "Null should return empty string");
        assertEquals(LlamaInference.sanitizeForJSON(""), "",   "Empty should return empty string");
    }

    @Test
    public void testFallbackSanitizationPlainTextUnchanged() {
        System.out.println("\n[TEST] sanitizeForJSON leaves plain text unchanged");
        String input  = "Simple text without special chars";
        String result = LlamaInference.sanitizeForJSON(input);
        System.out.println("Input:  " + input);
        System.out.println("Output: " + result);
        assertEquals(result, input, "Plain text with no special chars should be unchanged");
    }

    // ===== parseAndValidateJSON (no model needed) =====

    @Test
    public void testParseValidCompactJSON() {
        System.out.println("\n[TEST] parseAndValidateJSON — valid compact JSON passes through");
        String valid  = "{\"name\":\"Alice\",\"age\":\"30\"}";
        String result = sanitizeAction.parseAndValidateJSON(valid, "compact");
        System.out.println("Input:  " + valid);
        System.out.println("Output: " + result);
        assertNotNull(result);
        assertTrue(result.contains("Alice"));
        assertFalse(result.contains("\n"), "Compact output should have no newlines");
    }

    @Test
    public void testParseValidJSONPrettyFormat() {
        System.out.println("\n[TEST] parseAndValidateJSON — pretty formatting applied");
        String valid  = "{\"name\":\"Alice\",\"age\":\"30\"}";
        String result = sanitizeAction.parseAndValidateJSON(valid, "pretty");
        System.out.println("Output:\n" + result);
        assertTrue(result.contains("\n"), "Pretty output should contain newlines");
    }

    @Test
    public void testParseStripsMarkdownFence() {
        System.out.println("\n[TEST] parseAndValidateJSON — strips markdown code fence");
        String fenced = "```json\n{\"key\":\"value\"}\n```";
        String result = sanitizeAction.parseAndValidateJSON(fenced, "compact");
        System.out.println("Output: " + result);
        assertFalse(result.contains("```"), "Markdown fence should be removed");
        assertTrue(result.contains("key"));
    }

    @Test
    public void testParseStripsThinkBlocks() {
        System.out.println("\n[TEST] parseAndValidateJSON — strips DeepSeek think blocks");
        String withThink = "<think>Let me fix this JSON.</think>{\"result\":\"ok\"}";
        String result    = sanitizeAction.parseAndValidateJSON(withThink, "compact");
        System.out.println("Output: " + result);
        assertFalse(result.contains("<think>"), "Think block should be removed");
        assertTrue(result.contains("result"));
    }

    @Test(expectedExceptions = BotCommandException.class)
    public void testParseThrowsOnEmptyResponse() {
        System.out.println("\n[TEST] parseAndValidateJSON — throws on empty response");
        sanitizeAction.parseAndValidateJSON("", "compact");
    }

    @Test(expectedExceptions = BotCommandException.class)
    public void testParseThrowsOnNoJSONStructure() {
        System.out.println("\n[TEST] parseAndValidateJSON — throws when no JSON structure found");
        sanitizeAction.parseAndValidateJSON("No JSON here at all.", "compact");
    }

    @Test(expectedExceptions = BotCommandException.class)
    public void testParseThrowsOnStillInvalidJSON() {
        System.out.println("\n[TEST] parseAndValidateJSON — throws when JSON is still broken after extraction");
        // Deliberately malformed — missing closing quote and brace
        sanitizeAction.parseAndValidateJSON("{\"key\": \"unclosed", "compact");
    }

    // ===== Cross-platform path check =====

    @Test
    public void testModelCachePathIsAbsolute() {
        System.out.println("\n[TEST] Model cache path is absolute");
        Path cacheDir = Paths.get(System.getProperty("user.home"), "localAI");
        System.out.println("Cache dir: " + cacheDir.toAbsolutePath());
        assertTrue(cacheDir.isAbsolute(), "Model cache path should be absolute");
    }

    @Test
    public void testModelManagerSingleton() {
        System.out.println("\n[TEST] ModelManager is singleton and has all model types");
        ModelManager m1 = ModelManager.getInstance();
        ModelManager m2 = ModelManager.getInstance();
        assertSame(m1, m2, "ModelManager should be a singleton");

        for (ModelManager.ModelType type : ModelManager.ModelType.values()) {
            assertNotNull(m1.getModelPath(type),      "Model path should not be null for " + type.getId());
            assertNotNull(m1.getModelDirectory(type), "Model dir should not be null for " + type.getId());
            System.out.println("  " + type.getId() + " → " + m1.getModelPath(type));
        }
    }

    // ===== Full model inference tests (disabled — require model download) =====

    @Test(enabled = false)
    public void testRepairMalformedJSONWithModel() {
        System.out.println("\n[TEST] SanitizeJSON repairs broken JSON via model");

        // Has unescaped double quotes inside values and bare backslashes
        String brokenJson =
            "{\"summary\": \"User reported \"app crash\" on startup\", " +
            "\"path\": \"C:\\Users\\jsmith\\AppData\\crash.log\"}";

        System.out.println("Input:\n  " + brokenJson);
        System.out.println("(First run: model download + loading may take 1-3 minutes)\n");

        long start = System.currentTimeMillis();
        DictionaryValue result = sanitizeAction.execute(brokenJson, "compact", "qwen3-4b", 60.0, 0.0);
        long elapsed = System.currentTimeMillis() - start;

        String sanitized = ((StringValue) result.get("sanitized_json")).get();
        String status    = ((StringValue) result.get("status")).get();

        System.out.println("Status:    " + status);
        System.out.println("Completed: " + elapsed + "ms");
        System.out.println("Output:\n  " + sanitized);

        assertEquals(status, "success");
        assertNotNull(sanitized);
        assertFalse(sanitized.isEmpty());
    }

    @Test(enabled = false)
    public void testRepairJSONWithLiteralNewlines() {
        System.out.println("\n[TEST] SanitizeJSON repairs JSON with literal newlines in values");

        String brokenJson =
            "{\"notes\": \"Occurred after update.\nNeeds review.\tEscalate to Tier 2.\"}";

        System.out.println("Input: " + brokenJson);

        DictionaryValue result = sanitizeAction.execute(brokenJson, "compact", "qwen3-4b", 60.0, 0.0);
        String sanitized = ((StringValue) result.get("sanitized_json")).get();
        String status    = ((StringValue) result.get("status")).get();

        System.out.println("Status: " + status);
        System.out.println("Output: " + sanitized);

        assertEquals(status, "success");
        assertFalse(sanitized.contains("\n"), "Repaired JSON should not have unescaped newlines");
    }
}
