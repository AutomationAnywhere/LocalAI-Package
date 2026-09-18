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
 * Documents why gemma3-270m is excluded from TransformToJSON and
 * SanitizeJSON's model dropdowns. All three tests are disabled because the
 * model fails them, confirmed via live inference — not assumed:
 *
 *   - SanitizeJSON: given {"name": "John "The Rock" Smith", "age": 34,}
 *     (an unescaped nested quote + trailing comma), it returned valid JSON
 *     ({"name":"John","age":34.0}) but silently truncated the name value,
 *     dropping "The Rock" Smith" entirely — data loss, not just a
 *     formatting miss.
 *   - TransformToJSON (CSV): given two rows of name/age/city, it returned
 *     valid JSON with fabricated values — wrong ages, only first names,
 *     no cities at all. The output looked plausible but the data was
 *     invented, not extracted.
 *   - TransformToJSON (key-value): given Name/Invoice/Amount fields, it
 *     dropped Invoice and Amount and fabricated a field ("Age") that
 *     was never in the input.
 *
 * This is a materially worse failure mode than ClassifyText's (obviously
 * wrong category) or RedactPII's (input echoed unchanged, easy to spot) —
 * here the output is syntactically valid, plausible-looking JSON with
 * silently wrong or fabricated content, which is the failure mode most
 * likely to go unnoticed in a production bot. gemma3-270m is scoped to
 * the Prompt action only.
 */
public class TestGemma270MJSONTasks {

    private static final String MODEL = "gemma3-270m";
    private static final double TIMEOUT = 60.0;

    @AfterClass
    public void tearDown() {
        ModelManager.getInstance().shutdown();
    }

    @Test(enabled = false) // truncates values (data loss) — see class Javadoc
    public void testSanitizeBrokenJSONNotReliable() {
        SanitizeJSON action = new SanitizeJSON();
        String broken = "{\"name\": \"John \"The Rock\" Smith\", \"age\": 34,}";
        DictionaryValue result = action.execute(broken, "compact", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("sanitized_json")).get();
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        String name = obj.get("name").getAsString();
        assertTrue(name.contains("John") && name.contains("Smith") && name.contains("Rock"),
            "Expected full name value preserved (not truncated), got: " + name);
    }

    @Test(enabled = false) // fabricates values instead of extracting them — see class Javadoc
    public void testTransformCSVToJSONNotReliable() {
        TransformToJSON action = new TransformToJSON();
        String csv = "name,age,city\nJohn Smith,34,Boston\nJane Doe,29,Austin";
        DictionaryValue result = action.execute(csv, "csv", "compact", "array", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("json")).get();
        JsonElement parsed = JsonParser.parseString(json);
        String flat = json.toLowerCase();
        assertTrue(flat.contains("john smith") && flat.contains("boston"), "Expected first row's data present");
        assertTrue(flat.contains("jane doe") && flat.contains("austin"), "Expected second row's data present");
    }

    @Test(enabled = false) // drops real fields and fabricates one that wasn't in the input — see class Javadoc
    public void testTransformKeyValueToJSONNotReliable() {
        TransformToJSON action = new TransformToJSON();
        String kv = "Name: Acme Corp\nInvoice: 12345\nAmount: 1250.00";
        DictionaryValue result = action.execute(kv, "key-value", "compact", "object", MODEL, TIMEOUT, 0.0);
        String json = ((StringValue) result.get("json")).get();
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        assertTrue(obj.has("Invoice"), "Expected 'Invoice' field from input to be present, got: " + json);
        assertTrue(obj.has("Amount"), "Expected 'Amount' field from input to be present, got: " + json);
        assertFalse(obj.has("Age"), "Did not expect a fabricated 'Age' field (not in input), got: " + json);
    }
}
