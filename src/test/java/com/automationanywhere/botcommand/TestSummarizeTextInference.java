package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Real end-to-end inference test for SummarizeText — previously this action had
 * only prompt/parsing unit tests (TestSummarizeTextParsing), never a test that
 * drives it through the actual LlamaServerManager subprocess pipeline. Added
 * during the llama-server --host connectivity audit to close that gap.
 *
 * Uses deepseek-r1-1.5b (smallest model, likely already cached) to keep this
 * fast — this test is about validating the action's plumbing through a real
 * server round-trip, not summary quality from a specific model.
 */
public class TestSummarizeTextInference {

    private static final String MODEL = "deepseek-r1-1.5b";
    private static final double TIMEOUT = 120.0;

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance();
        System.out.println("SummarizeText inference test completed");
    }

    @Test
    public void testSummarizeTextRealInference() {
        SummarizeText action = new SummarizeText();
        String input = "The quarterly report shows revenue increased 12% year over year, "
            + "driven primarily by growth in the enterprise segment. Operating costs "
            + "rose slightly due to increased headcount in engineering. The board "
            + "approved a new expansion into two additional markets for next quarter.";

        DictionaryValue result = action.execute(input, "short", null, MODEL, TIMEOUT, 0.0);

        assertNotNull(result, "Result should not be null");
        String summary = ((StringValue) result.get("summary")).get();
        assertNotNull(summary, "summary should not be null");
        assertFalse(summary.isEmpty(), "summary should not be empty");

        System.out.println("Input:   " + input);
        System.out.println("Summary: " + summary);
    }
}
