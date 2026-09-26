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

    // ── Context size: sized to the request ────────────────────────────────────

    private static final long GB = 1024L * 1024 * 1024;

    @Test
    public void testShortPromptGetsMinimumContext() {
        assertEquals(LlamaServerManager.contextSizeFor(300, 32768, 16384), LlamaServerManager.MIN_CONTEXT);
        assertEquals(LlamaServerManager.contextSizeFor(0, 32768, 16384), LlamaServerManager.MIN_CONTEXT);
    }

    @Test
    public void testContextGrowsToSmallestFittingPowerOfTwo() {
        // 4096 * 0.9 = 3686 budget, so 3687 needs 8192
        assertEquals(LlamaServerManager.contextSizeFor(3686, 131072, 32768), 4096);
        assertEquals(LlamaServerManager.contextSizeFor(3687, 131072, 32768), 8192);
        assertEquals(LlamaServerManager.contextSizeFor(10000, 131072, 32768), 16384);
    }

    @Test
    public void testChosenContextAlwaysFitsWhenUnderLimits() {
        for (int required = 0; required <= 29000; required += 250) {
            int ctx = LlamaServerManager.contextSizeFor(required, 131072, 32768);
            assertTrue(LlamaServerManager.contextBudget(ctx) >= required, "required=" + required + " ctx=" + ctx);
        }
    }

    @Test
    public void testContextStopsAtMachineCeiling() {
        assertEquals(LlamaServerManager.contextSizeFor(100000, 131072, 16384), 16384);
    }

    @Test
    public void testContextStopsAtModelMaximum() {
        assertEquals(LlamaServerManager.contextSizeFor(100000, 8192, 32768), 8192);
        // ceiling that isn't a power of two is still honoured exactly
        assertEquals(LlamaServerManager.contextSizeFor(100000, 131072, 12000), 12000);
    }

    @Test
    public void testCeilingFollowsInstalledRam() {
        assertEquals(LlamaServerManager.defaultMaxContext(8 * GB), 8192);
        assertEquals(LlamaServerManager.defaultMaxContext(12 * GB), 16384);
        assertEquals(LlamaServerManager.defaultMaxContext(16 * GB), 32768);
    }

    @Test
    public void testUnknownRamIsTreatedAsSmallRunner() {
        assertEquals(LlamaServerManager.defaultMaxContext(-1), 8192);
    }
}
