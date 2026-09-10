package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Real end-to-end inference test for RedactPII — previously this action had
 * only prompt/parsing unit tests (TestRedactPIIParsing), never a test that
 * drives it through the actual LlamaServerManager subprocess pipeline. Added
 * during the llama-server --host connectivity audit to close that gap.
 *
 * Uses deepseek-r1-1.5b (smallest model, likely already cached) to keep this
 * fast — this test is about validating the action's plumbing through a real
 * server round-trip, not redaction quality from a specific model.
 */
public class TestRedactPIIInference {

    private static final String MODEL = "deepseek-r1-1.5b";
    private static final double TIMEOUT = 120.0;

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance();
        System.out.println("RedactPII inference test completed");
    }

    @Test
    public void testRedactPIIRealInference() {
        RedactPII action = new RedactPII();
        String input = "Contact John Smith at john.smith@example.com or 555-123-4567.";

        DictionaryValue result = action.execute(input, "all", "[REDACTED]", MODEL, TIMEOUT);

        assertNotNull(result, "Result should not be null");
        String redacted = ((StringValue) result.get("redacted_text")).get();
        assertNotNull(redacted, "redacted_text should not be null");
        assertFalse(redacted.isEmpty(), "redacted_text should not be empty");

        System.out.println("Input:    " + input);
        System.out.println("Redacted: " + redacted);
    }
}
