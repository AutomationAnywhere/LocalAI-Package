package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Real end-to-end inference tests for Gemma 3 270M (Q4_K_M, ~253MB) — the
 * smallest model in the lineup, added for narrow single-purpose tasks.
 *
 * Verified reliable for Prompt (simple factual Q&A). NOT reliable for
 * ClassifyText or RedactPII, so gemma3-270m is excluded from both of their
 * model dropdowns entirely (see ClassifyText.java / RedactPII.java) rather
 * than offered as an unreliable option:
 *   - ClassifyText: across 5 repeated trials against 4 categories
 *     (invoice/receipt/contract/report), the model defaulted to "Receipt"
 *     as a catch-all answer — 0/5 correct on contract and report, 2/5 on
 *     invoice, 5/5 on receipt (which just matches its bias).
 *   - RedactPII: returned the input completely unmodified, no entities
 *     redacted at all.
 * It was not exposed in ExtractData/SanitizeJSON/NormalizeAndStandardize/
 * SummarizeText/TransformToJSON either — those require the same kind of
 * reliable structured instruction-following that failed above, and weren't
 * separately verified as working.
 */
public class TestGemma270MInference {

    private static final String MODEL = "gemma3-270m";
    private static final double TIMEOUT = 60.0;

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
    }

    @Test
    public void testBasicPrompt() {
        Prompt action = new Prompt();
        DictionaryValue result = action.execute("Q: What is the capital of Japan? A:", MODEL, TIMEOUT, 0.1);
        String response = ((StringValue) result.get("response")).get();
        System.out.println("[Prompt] response: " + response);
        assertNotNull(response);
        assertFalse(response.trim().isEmpty());
    }

    @Test(enabled = false) // unreliable: defaults to "Receipt" regardless of input — see class Javadoc
    public void testClassificationNotReliable() {
        ClassifyText action = new ClassifyText();
        String text = "This Agreement is entered into by and between Party A and Party B, "
            + "effective as of the date of signing, subject to the terms herein.";
        DictionaryValue result = action.execute(
            text, "invoice, receipt, contract, report", MODEL, false, false, TIMEOUT);
        String category = ((StringValue) result.get("category")).get();
        assertTrue(category.toLowerCase().contains("contract"),
            "Expected 'contract' in category, got: " + category);
    }

    @Test(enabled = false) // gemma3-270m returns input unmodified; excluded from RedactPII's model dropdown
    public void testRedactPIINotReliable() {
        RedactPII action = new RedactPII();
        String input = "Contact John Smith at john.smith@example.com or 555-123-4567.";
        DictionaryValue result = action.execute(input, "all", "[REDACTED]", MODEL, TIMEOUT);
        String redacted = ((StringValue) result.get("redacted_text")).get();
        assertFalse(redacted.contains("john.smith@example.com"), "Email should be redacted");
        assertFalse(redacted.contains("555-123-4567"), "Phone should be redacted");
        assertFalse(redacted.contains("John Smith"), "Name should be redacted");
    }
}
