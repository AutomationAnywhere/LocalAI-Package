package com.automationanywhere.botcommand;

import com.automationanywhere.botcommand.data.impl.DictionaryValue;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.botcommand.utils.ModelManager;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Real end-to-end inference tests for Gemma 3 1B (Q4_K_M, ~769MB, 32K ctx) —
 * added after gemma3-270m turned out too small for anything but Prompt.
 * Verified reliable across Prompt, ClassifyText (5/5 correct across all 4
 * categories, repeated), RedactPII, TransformToJSON, SanitizeJSON, and
 * SummarizeText. Excluded from ExtractData and NormalizeAndStandardize (see
 * their model dropdowns for the specific failures found in evaluation).
 */
public class TestGemma3_1BInference {

    private static final String MODEL = "gemma3-1b";
    private static final double TIMEOUT = 60.0;

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
    }

    @Test
    public void testBasicPrompt() {
        Prompt action = new Prompt();
        DictionaryValue result = action.execute("Q: What is the capital of Japan? A:", MODEL, TIMEOUT, 0.1, 0.0);
        String response = ((StringValue) result.get("response")).get();
        System.out.println("[Prompt] response: " + response);
        assertNotNull(response);
        assertTrue(response.toLowerCase().contains("tokyo"), "Expected 'Tokyo' in response, got: " + response);
    }

    @Test
    public void testClassificationInvoice() {
        assertClassification("INVOICE #12345\nDate: 2026-01-15\nBill To: Acme Corp\nAmount Due: $1,250.00", "invoice");
    }

    @Test
    public void testClassificationReceipt() {
        assertClassification("Thank you for your purchase! Receipt #9981. Item: Wireless Mouse. Total paid: $24.99.", "receipt");
    }

    @Test
    public void testClassificationContract() {
        assertClassification("This Agreement is entered into by and between Party A and Party B, "
            + "effective as of the date of signing, subject to the terms herein.", "contract");
    }

    @Test
    public void testClassificationReport() {
        assertClassification("Q3 Sales Report: Revenue grew 12% year over year, driven by strong "
            + "performance in the enterprise segment.", "report");
    }

    private void assertClassification(String text, String expectedCategory) {
        ClassifyText action = new ClassifyText();
        DictionaryValue result = action.execute(text, "invoice, receipt, contract, report", MODEL, false, false, TIMEOUT, 0.0);
        String category = ((StringValue) result.get("category")).get();
        System.out.println("[ClassifyText] expected=" + expectedCategory + " -> got: " + category);
        assertTrue(category.toLowerCase().contains(expectedCategory),
            "Expected '" + expectedCategory + "' in category, got: " + category);
    }

    @Test
    public void testRedactPII() {
        RedactPII action = new RedactPII();
        String input = "Contact John Smith at john.smith@example.com or 555-123-4567.";
        DictionaryValue result = action.execute(input, "all", "[REDACTED]", MODEL, TIMEOUT, 0.0);
        String redacted = ((StringValue) result.get("redacted_text")).get();
        System.out.println("[RedactPII] input:    " + input);
        System.out.println("[RedactPII] redacted: " + redacted);
        assertFalse(redacted.contains("john.smith@example.com"), "Email should be redacted");
        assertFalse(redacted.contains("555-123-4567"), "Phone should be redacted");
        assertFalse(redacted.contains("John Smith"), "Name should be redacted");
    }

    @Test
    public void testTransformCSVToJSON() {
        TransformToJSON action = new TransformToJSON();
        String csv = "name,age,city\nJohn Smith,34,Boston\nJane Doe,29,Austin";
        DictionaryValue result = action.execute(csv, "csv", "compact", "array", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("json")).get();
        System.out.println("[TransformToJSON:csv] output: " + json);
        JsonElement parsed = JsonParser.parseString(json);
        assertTrue(parsed.isJsonArray());
        String flat = json.toLowerCase();
        assertTrue(flat.contains("john smith") && flat.contains("boston"), "Expected first row's data present");
        assertTrue(flat.contains("jane doe") && flat.contains("austin"), "Expected second row's data present");
    }

    @Test
    public void testTransformKeyValueToJSON() {
        TransformToJSON action = new TransformToJSON();
        String kv = "Name: Acme Corp\nInvoice: 12345\nAmount: 1250.00";
        DictionaryValue result = action.execute(kv, "key-value", "compact", "object", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("json")).get();
        System.out.println("[TransformToJSON:key-value] output: " + json);
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        assertTrue(obj.has("Invoice"), "Expected 'Invoice' field, got: " + json);
        assertTrue(obj.has("Amount"), "Expected 'Amount' field, got: " + json);
    }

    @Test
    public void testSanitizeBrokenJSON() {
        SanitizeJSON action = new SanitizeJSON();
        String broken = "{\"name\": \"John \"The Rock\" Smith\", \"age\": 34,}";
        DictionaryValue result = action.execute(broken, "compact", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("sanitized_json")).get();
        System.out.println("[SanitizeJSON] input:  " + broken);
        System.out.println("[SanitizeJSON] output: " + json);
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        String name = obj.get("name").getAsString();
        assertTrue(name.contains("John") && name.contains("Smith") && name.contains("Rock"),
            "Expected full name value preserved (not truncated), got: " + name);
    }

    @Test
    public void testSummarize() {
        SummarizeText action = new SummarizeText();
        String text = "The quarterly earnings call revealed that revenue grew 15% year-over-year, "
            + "driven primarily by strong enterprise subscription renewals. However, operating "
            + "costs also increased due to expanded headcount in engineering and sales. "
            + "Management reiterated full-year guidance and highlighted plans to launch two new "
            + "product lines in the next two quarters.";
        DictionaryValue result = action.execute(text, "short", null, MODEL, TIMEOUT, 0.0);
        String summary = ((StringValue) result.get("summary")).get();
        System.out.println("[SummarizeText] summary: " + summary);
        assertNotNull(summary);
        assertTrue(summary.toLowerCase().contains("revenue") || summary.toLowerCase().contains("earnings"),
            "Expected summary to reference the topic, got: " + summary);
    }

    @Test(enabled = false) // echoes literal "FIELDNAME" placeholder, breaking field-name matching — see ExtractData.java
    public void testExtractDataNotReliable() {
        ExtractData action = new ExtractData();
        String text = "INVOICE #12345\nDate: 2026-01-15\nBill To: Acme Corp\nAmount Due: $1,250.00";
        String fields = "invoice_number: the invoice ID\ntotal_amount: the total amount due\nvendor_name: who the bill is to";
        DictionaryValue result = action.execute(text, fields, MODEL, TIMEOUT, 0.0);
        assertFalse(ExtractData.NOT_FOUND.equals(((StringValue) result.get("invoice_number")).get()));
    }

    @Test(enabled = false) // phone normalization doesn't reliably comply with requested format — see NormalizeAndStandardize.java
    public void testNormalizePhoneNotReliable() {
        NormalizeAndStandardize action = new NormalizeAndStandardize();
        DictionaryValue result = action.execute("Call me at (555) 123-4567", "phone", "E.164", MODEL, true, TIMEOUT, 0.0);
        String normalized = ((StringValue) result.get("result")).get();
        assertTrue(normalized.startsWith("+1"), "Expected E.164 format with +1 country code, got: " + normalized);
    }
}
