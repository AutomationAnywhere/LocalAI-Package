package com.automationanywhere.botcommand.utils;

import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Unit tests for llama-server sizing defaults (thread count, context cap).
 * No model download or llama-server binary required.
 */
public class TestLlamaServerManagerConfig {

    // ── Thread count ──────────────────────────────────────────────────────────

    @Test
    public void testSmallRunnersUseEveryCore() {
        assertEquals(LlamaServerManager.defaultThreadCount(1), 1);
        assertEquals(LlamaServerManager.defaultThreadCount(2), 2);
        assertEquals(LlamaServerManager.defaultThreadCount(4), 4);
    }

    @Test
    public void testLargerMachinesLeaveHeadroom() {
        assertEquals(LlamaServerManager.defaultThreadCount(6), 4);
        assertEquals(LlamaServerManager.defaultThreadCount(10), 8);
    }

    @Test
    public void testZeroCoresReportedStillGetsOneThread() {
        assertEquals(LlamaServerManager.defaultThreadCount(0), 1);
    }

    // ── Context size ──────────────────────────────────────────────────────────

    @Test
    public void testLargeWindowModelsAreCappedByDefault() {
        assertEquals(LlamaServerManager.effectiveContextSize(131072, null), LlamaServerManager.DEFAULT_CONTEXT_CAP);
        assertEquals(LlamaServerManager.effectiveContextSize(32768, null), LlamaServerManager.DEFAULT_CONTEXT_CAP);
    }

    @Test
    public void testSmallWindowModelKeepsItsOwnWindow() {
        assertEquals(LlamaServerManager.effectiveContextSize(4096, null), 4096);
    }

    @Test
    public void testOverrideRaisesCap() {
        assertEquals(LlamaServerManager.effectiveContextSize(131072, 32768), 32768);
    }

    @Test
    public void testOverrideIsClampedToModelMaximum() {
        assertEquals(LlamaServerManager.effectiveContextSize(8192, 65536), 8192);
    }
}
