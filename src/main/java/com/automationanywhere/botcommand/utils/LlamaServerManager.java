package com.automationanywhere.botcommand.utils;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import okhttp3.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;

/**
 * Manages a llama-server subprocess for local model inference.
 *
 * Security: each server instance is started with a randomly-generated UUID
 * API key (--api-key flag). All HTTP requests include the key as an
 * "Authorization: Bearer" header so other local processes cannot piggyback
 * on the inference endpoint.
 *
 * Thread safety: mutable state (port, apiKey, serverProcess, currentModelId)
 * is declared volatile.  complete() captures port and apiKey under a brief
 * synchronized block, then releases the lock before the long HTTP call so
 * that stop()/ensureModelLoaded() are not blocked for the full inference
 * duration.  ensureModelLoaded() and stop() are fully synchronized.
 */
public class LlamaServerManager {
    private static final Logger logger = LogManager.getLogger(LlamaServerManager.class);
    private static final MediaType JSON_TYPE = MediaType.parse("application/json; charset=utf-8");

    // The interface llama-server binds to AND the interface our HTTP client connects to.
    // Previously neither side was pinned: llama-server used its own internal default bind
    // address, and the client separately hardcoded "127.0.0.1" — relying on those two
    // independently-chosen values happening to agree. In VM/VDI environments with network
    // virtualization, VPN split-tunnel drivers, or endpoint security that intercepts
    // loopback traffic, they can diverge: the server comes up fine on a real, working
    // address that isn't the one the client polls, producing "llama-server did not become
    // ready within 120s" even though the process started successfully. Pinning both sides
    // to the same explicit value removes that ambiguity entirely.
    //
    // Default "127.0.0.1" needs zero configuration and works for the overwhelming majority
    // of installs. For the rare environment where loopback genuinely does not route between
    // the bot process and its own child llama-server process, override with:
    //   - LOCALAI_SERVER_HOST environment variable (preferred — set once via Windows
    //     System Properties > Environment Variables, no Bot Agent config file editing,
    //     no JVM flag syntax to get wrong), or
    //   - -Dlocalai.server.host=<ip> JVM system property, if env var isn't practical
    // Env var takes priority when both are set (it's the friendlier path for a non-engineer
    // to set correctly), falling back to the system property, falling back to 127.0.0.1.
    private static final String SERVER_HOST = resolveServerHost();

    private static String resolveServerHost() {
        String fromEnv = System.getenv("LOCALAI_SERVER_HOST");
        if (fromEnv != null && !fromEnv.trim().isEmpty()) {
            return fromEnv.trim();
        }
        return System.getProperty("localai.server.host", "127.0.0.1");
    }

    // By default llama-server binds a fresh OS-assigned ephemeral port every run
    // (ServerSocket(0) — see findFreePort()). That's the right default: zero
    // configuration, no port ever collides. But some locked-down VM/VDI images
    // only allow firewall/network policy rules for a specific, known port (a
    // different random port every run can't be allowlisted at all) — even on
    // loopback, some enterprise endpoint security filters by port regardless of
    // address. For that case, LOCALAI_SERVER_PORT (or -Dlocalai.server.port) pins
    // a fixed port instead of picking a random one. Same env-var-first, then
    // system-property, then "use dynamic" precedence as SERVER_HOST above.
    private static final Integer FIXED_SERVER_PORT = resolveServerPort();

    private static Integer resolveServerPort() {
        String raw = System.getenv("LOCALAI_SERVER_PORT");
        if (raw == null || raw.trim().isEmpty()) {
            raw = System.getProperty("localai.server.port");
        }
        if (raw == null || raw.trim().isEmpty()) {
            return null; // no override — use a dynamic ephemeral port
        }
        try {
            int p = Integer.parseInt(raw.trim());
            if (p < 1 || p > 65535) {
                logger.warn("LOCALAI_SERVER_PORT/localai.server.port value '{}' is out of range " +
                    "(1-65535) — falling back to a dynamic port", raw);
                return null;
            }
            return p;
        } catch (NumberFormatException e) {
            logger.warn("LOCALAI_SERVER_PORT/localai.server.port value '{}' is not a valid integer " +
                "— falling back to a dynamic port", raw);
            return null;
        }
    }

    // Number of CPU threads llama-server uses for inference (-t / --threads).
    // Previously no -t flag was passed at all, leaving llama-server to
    // auto-detect. On machines with a performance/efficiency core split
    // (e.g. Apple Silicon, many recent Intel/AMD laptop chips), that
    // auto-detect logic already avoids using every logical core, but still
    // leaves real throughput on the table: benchmarked on an Apple M4
    // (4P+6E cores, 10 logical) with llama-server b9481, the auto-detected
    // thread count generated Qwen3-4B-Q4_K_M output at ~26 tok/s, while
    // explicitly setting threads = logical cores - 2 (here, 8) reached
    // ~34-35 tok/s — a ~30% generation speedup, with prompt processing
    // throughput also improved (~110 -> ~137 tok/s). "logical cores - 2"
    // is a portable heuristic (no cross-platform P/E-core detection
    // available in pure Java) that leaves headroom for the OS and the bot
    // runner process itself; it is not guaranteed optimal on every CPU
    // topology, hence the override below for hardware where it isn't.
    //
    // Small machines are the exception: typical Bot Runners are 2-4 vCPU VMs,
    // where "minus 2" left 1-2 threads and roughly halved throughput. The bot
    // is blocked waiting on the inference call anyway, so on <= 4 logical
    // cores every core goes to llama-server. Benchmarked on a 4 vCPU VM with a
    // Qwen3-4B-shaped Q4_K_M model: 2 threads -> 4 threads took prompt
    // processing from ~19 to ~37 tok/s and generation from ~3.3 to ~5.9 tok/s.
    private static final int SERVER_THREADS = resolveServerThreads();

    private static int resolveServerThreads() {
        Integer override = parsePositiveInt(System.getenv("LOCALAI_SERVER_THREADS"));
        if (override == null) override = parsePositiveInt(System.getProperty("localai.server.threads"));
        return override != null ? override : defaultThreadCount(Runtime.getRuntime().availableProcessors());
    }

    static int defaultThreadCount(int logicalCores) {
        if (logicalCores <= 4) return Math.max(1, logicalCores);
        return logicalCores - 2;
    }

    // Context window (-c) is sized to the request, not the model's maximum.
    // The KV cache is allocated (and zeroed) in full for -c at load time, so
    // running every model at its advertised window (32K-128K) made the default
    // qwen3-4b reserve ~7.1GB RSS and take ~33s to load on a 4 vCPU VM, even
    // for a one-line classification. Measured peak RSS for qwen3-4b:
    // 8K ~3.7GB, 16K ~4.8GB, 32K ~7.1GB.
    //
    // The server starts at the smallest power of two >= MIN_CONTEXT whose
    // budget fits the first request, and restarts one size up only when a
    // later request doesn't fit. It never shrinks, so a bot looping over
    // mixed-length documents reloads at most a couple of times rather than
    // on every call. Growth stops at a ceiling picked from physical RAM (see
    // defaultMaxContext) so one long input can't push the runner into swap;
    // past that, complete() fails fast with an actionable error. Override the
    // ceiling with LOCALAI_CONTEXT_SIZE or -Dlocalai.context.size. It is
    // always clamped to the model's own maximum.
    static final int MIN_CONTEXT = 4096;
    private static final int MAX_CONTEXT = resolveMaxContext();

    private static int resolveMaxContext() {
        Integer override = parsePositiveInt(System.getenv("LOCALAI_CONTEXT_SIZE"));
        if (override == null) override = parsePositiveInt(System.getProperty("localai.context.size"));
        return override != null ? override : defaultMaxContext(totalPhysicalMemoryBytes());
    }

    static int defaultMaxContext(long totalRamBytes) {
        long gb = 1024L * 1024 * 1024;
        if (totalRamBytes <= 0 || totalRamBytes < 10 * gb) return 8192;  // unknown or ~8GB runner
        if (totalRamBytes < 15 * gb) return 16384;                       // ~12GB runner
        return 32768;
    }

    /** Smallest power of two >= MIN_CONTEXT whose budget fits requiredTokens, clamped to the limits. */
    static int contextSizeFor(int requiredTokens, int modelContextWindow, int maxContext) {
        int limit = Math.min(modelContextWindow, maxContext);
        int size = MIN_CONTEXT;
        while (size < limit && contextBudget(size) < requiredTokens) size *= 2;
        return Math.min(size, limit);
    }

    /** Tokens usable in a context of {@code ctx}: 10% margin for the ~4 chars/token estimate. */
    static int contextBudget(int ctx) {
        return (int) (ctx * 0.9);
    }

    @SuppressWarnings("deprecation") // getTotalPhysicalMemorySize: still present, and the only name on Java 11
    private static long totalPhysicalMemoryBytes() {
        try {
            java.lang.management.OperatingSystemMXBean os = java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            if (os instanceof com.sun.management.OperatingSystemMXBean) {
                return ((com.sun.management.OperatingSystemMXBean) os).getTotalPhysicalMemorySize();
            }
        } catch (Throwable t) {
            logger.debug("Could not read physical memory size: {}", t.toString());
        }
        return -1;
    }

    private static Integer parsePositiveInt(String raw) {
        if (raw == null || raw.trim().isEmpty()) return null;
        try {
            int v = Integer.parseInt(raw.trim());
            return v > 0 ? v : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static volatile LlamaServerManager instance;

    // Volatile: writes happen inside synchronized methods; reads in complete()
    // and isRunning() happen outside, so volatile is needed for visibility.
    private volatile Process serverProcess;
    private volatile int     port = -1;
    private volatile String  currentModelId;
    private volatile String  apiKey; // random UUID, regenerated on each server start
    private volatile int     currentContextWindow = -1; // -c value the running server was started with

    // Generous base timeouts for health-check polling during model load.
    // Per-call inference timeouts are applied in complete() via newBuilder().
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
        .connectTimeout(Duration.ofSeconds(10))
        .readTimeout(Duration.ofMinutes(5))
        .callTimeout(Duration.ofMinutes(5))
        .build();
    private final Gson gson = new Gson();

    private LlamaServerManager() {
        Runtime.getRuntime().addShutdownHook(new Thread(this::stop, "llama-server-shutdown"));
    }

    public static LlamaServerManager getInstance() {
        if (instance == null) {
            synchronized (LlamaServerManager.class) {
                if (instance == null) {
                    instance = new LlamaServerManager();
                }
            }
        }
        return instance;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Ensure llama-server is running with the given model.
     * Fast-path: reuses a healthy server when the model already matches (and,
     * when a specific port was requested, only when it's already bound there).
     * Slow-path: stops the current server and starts a new one.
     *
     * @param portOverride a specific port to bind to (e.g. from an action's
     *                     optional "Server Port" field), or null/&lt;=0 to use
     *                     the LOCALAI_SERVER_PORT env var / system property if
     *                     set, falling back to an OS-assigned dynamic port —
     *                     the original zero-config behavior. Passing this per
     *                     call is friendlier for locked-down network policies
     *                     that require a fixed, allowlisted port than editing
     *                     an environment variable, since it's just a bot
     *                     action field with a default that changes nothing
     *                     for everyone else.
     */
    public void ensureModelLoaded(ModelManager.ModelType modelType, Integer portOverride) throws Exception {
        ensureModelLoaded(modelType, portOverride, 0);
    }

    /**
     * As above, but also guarantees the context window can hold
     * {@code requiredTokens} (estimated prompt + max output tokens), restarting
     * one size up if the running server is too small — see MIN_CONTEXT.
     */
    public synchronized void ensureModelLoaded(ModelManager.ModelType modelType, Integer portOverride,
                                               int requiredTokens) throws Exception {
        int ctx = contextSizeFor(requiredTokens, modelType.getContextWindow(), MAX_CONTEXT);
        boolean sameModel = isRunning() && modelType.getId().equals(currentModelId);
        boolean portMismatch = portOverride != null && portOverride > 0 && isRunning() && port != portOverride;
        if (sameModel && !portMismatch && currentContextWindow >= ctx) {
            logger.debug("Reusing running server for model: {} (ctx {})", currentModelId, currentContextWindow);
            return;
        }
        if (sameModel) {
            ctx = Math.max(ctx, currentContextWindow); // never shrink on a same-model restart
            logger.info("Restarting llama-server for model {} with a larger context: {} -> {}",
                currentModelId, currentContextWindow, ctx);
        }

        stopInternal();

        Path modelPath = ModelManager.getInstance().getModelPath(modelType);
        if (!Files.exists(modelPath)) {
            throw new RuntimeException(
                "Model file not found: " + modelPath + ". Run Validate Device first.");
        }

        LlamaBinaryManager.ensureInstalled();
        port = (portOverride != null && portOverride > 0) ? portOverride
             : (FIXED_SERVER_PORT != null ? FIXED_SERVER_PORT : findFreePort());

        // startServer() may throw (e.g. model-load timeout). If it does,
        // waitForReady() internally calls stopInternal(), resetting port/apiKey/
        // serverProcess before re-throwing — leaving no stale state.
        startServer(modelPath, modelType, ctx);

        // Mark the model as loaded only after the server is confirmed healthy.
        currentModelId = modelType.getId();
    }

    private void startServer(Path modelPath, ModelManager.ModelType modelType, int ctx) throws Exception {
        Path serverBin = LlamaBinaryManager.getLlamaServerPath();
        String os = System.getProperty("os.name", "").toLowerCase();
        boolean isWindows = os.contains("windows");

        String absModelPath = modelPath.toAbsolutePath().toString();
        if (isWindows) absModelPath = absModelPath.replace('\\', '/');

        this.currentContextWindow = ctx; // sized by ensureModelLoaded — see MIN_CONTEXT

        // Generate a fresh random API key for this server instance.
        this.apiKey = UUID.randomUUID().toString();

        List<String> cmd = new ArrayList<>(Arrays.asList(
            serverBin.toString(),
            "-m",        absModelPath,
            "--host",    SERVER_HOST, // pin bind address — see SERVER_HOST comment above
            "--port",    String.valueOf(port),
            "-ngl",      "0",
            "-c",        String.valueOf(ctx),
            "-np",       "1",
            "-t",        String.valueOf(SERVER_THREADS),
            // Disable llama-server's host-RAM prompt cache (default 8192 MiB).
            // complete() sends cache_prompt=false, so it is never read, but the
            // server still saves each distinct prompt's KV state into it:
            // measured +~160MB RSS per distinct ~1200-token request, growing
            // toward 8GB over a long-running bot — the whole RAM of a runner.
            "--cache-ram", "0",
            "--api-key", this.apiKey   // reject requests that lack this key
        ));

        if (isWindows) cmd.add("--no-mmap");

        logger.info("Starting llama-server: model={}, host={}, port={}, ctx={}, threads={}",
            modelType.getId(), SERVER_HOST, port, ctx, SERVER_THREADS);

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.directory(serverBin.getParent().toFile()); // cwd=binDir for DLL resolution on Windows

        // ProcessBuilder.Redirect.to() truncates the log on each new server start,
        // keeping the file bounded to the output of a single session.
        Path logFile = ModelManager.getModelCacheDir().resolve("llama-server.log");
        pb.redirectOutput(ProcessBuilder.Redirect.to(logFile.toFile()));
        pb.redirectError(ProcessBuilder.Redirect.to(logFile.toFile()));

        serverProcess = pb.start();
        logger.info("llama-server started (PID {}). Log: {}", serverProcess.pid(), logFile);

        waitForReady(120_000);
    }

    private void waitForReady(long timeoutMs) throws Exception {
        String healthUrl = "http://" + SERVER_HOST + ":" + port + "/health";
        long deadline = System.currentTimeMillis() + timeoutMs;

        logger.info("Waiting for llama-server on {}:{} (up to {}s)...", SERVER_HOST, port, timeoutMs / 1000);

        // Track the last connection failure so a full timeout can report *why* the
        // health check never succeeded (e.g. connection refused vs. reset vs. timed
        // out) instead of just "did not become ready" — this was previously
        // indistinguishable from a genuine slow model load.
        IOException lastConnectFailure = null;
        while (System.currentTimeMillis() < deadline) {
            // Snapshot the volatile field once per iteration to avoid a race
            // between two reads of the same field within one loop body.
            Process proc = serverProcess;
            if (proc == null || !proc.isAlive()) {
                Path log = ModelManager.getModelCacheDir().resolve("llama-server.log");
                String tail = readLogTail(log, 20);
                stopInternal();
                throw new RuntimeException(
                    "llama-server exited unexpectedly. Last log lines:\n" + tail);
            }

            try {
                Request req = new Request.Builder()
                    .url(healthUrl)
                    .header("Authorization", "Bearer " + apiKey)
                    .get()
                    .build();
                try (Response resp = httpClient.newCall(req).execute()) {
                    if (resp.code() == 200) {
                        logger.info("llama-server ready on {}:{}", SERVER_HOST, port);
                        return;
                    }
                }
            } catch (IOException e) {
                // Not accepting connections yet — keep polling, but remember why
                // in case we never succeed and need to report it below.
                lastConnectFailure = e;
            }

            Thread.sleep(500);
        }

        Path log = ModelManager.getModelCacheDir().resolve("llama-server.log");
        String tail = readLogTail(log, 20);
        stopInternal();
        throw new RuntimeException(
            "llama-server did not become ready within " + (timeoutMs / 1000) + "s "
            + "(process was alive, health check against " + healthUrl + " never returned 200). "
            + "Last connection error: " + (lastConnectFailure != null ? lastConnectFailure.toString() : "none — got non-200 responses only") + ". "
            + "If this environment's network setup means " + SERVER_HOST + ":" + port + " does not route to "
            + "the bot's own child processes (e.g. some VM/VDI network virtualization, VPN split-tunnel "
            + "software, or a firewall/endpoint policy that blocks a random ephemeral port like " + port + "), "
            + "set the LOCALAI_SERVER_HOST environment variable to a working address and/or "
            + "LOCALAI_SERVER_PORT to a fixed port your network policy allows, then restart the Bot Agent "
            + "(or -Dlocalai.server.host / -Dlocalai.server.port as JVM arguments if you cannot set env vars). "
            + "Last server log lines:\n" + tail);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Inference
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Run a completion against the loaded model (no grammar constraint).
     * Delegates to the grammar-aware overload with {@code grammar = null}.
     */
    public String complete(String prompt, int maxTokens, float temperature,
                           String[] stopSequences, int timeoutSecs) throws Exception {
        return complete(prompt, maxTokens, temperature, stopSequences, timeoutSecs, null);
    }

    /**
     * Run a completion against the loaded model with an optional GBNF grammar constraint.
     *
     * When {@code grammar} is non-null, llama-server constrains token sampling at every
     * step so the output can only form strings that match the grammar.  For JSON grammars
     * this means the model physically cannot produce malformed JSON, preamble text, or
     * markdown fences — regardless of model size or quantization level.
     *
     * Port and apiKey are captured under a brief synchronized block, then the
     * lock is released before the HTTP call.  This avoids blocking
     * stop()/ensureModelLoaded() for the full 30-120 s inference window while
     * still preventing a race where stopInternal() zeroes {@code port} to -1
     * between the isRunning() guard and the URL construction below.
     *
     * @param grammar optional GBNF grammar string (see {@link JsonGrammar}); null = unconstrained
     */
    public String complete(String prompt, int maxTokens, float temperature,
                           String[] stopSequences, int timeoutSecs, String grammar) throws Exception {
        final int    localPort;
        final String localApiKey;
        final int    localCtx;
        final String localModelId;
        synchronized (this) {
            if (!isRunning()) {
                throw new RuntimeException(
                    "llama-server is not running. Call ensureModelLoaded() first.");
            }
            localPort    = this.port;
            localApiKey  = this.apiKey;
            localCtx     = this.currentContextWindow;
            localModelId = this.currentModelId;
        }

        // Fail fast, locally, with an actionable message instead of letting
        // llama-server reject or truncate the request. ensureModelLoaded() grows
        // the context to fit each request, so this only trips once a request
        // needs more than MAX_CONTEXT (or the model's own maximum), where the
        // server-side error would be an opaque HTTP failure. Token count is
        // estimated (~4 chars per token is a standard rough approximation for
        // English text with these tokenizers) — not exact, so contextBudget()
        // keeps a 10% safety margin.
        int estimatedPromptTokens = estimateTokenCount(prompt);
        int budget = contextBudget(localCtx);
        if (estimatedPromptTokens + maxTokens > budget) {
            throw new RuntimeException(String.format(
                "Prompt (~%d tokens, estimated) + max tokens (%d) exceeds the safe budget "
                + "(%d of %d context tokens) for model '%s'. Reduce the prompt length or max "
                + "tokens. If the model supports a larger window, you can raise this machine's "
                + "context ceiling (currently %d) with the "
                + "LOCALAI_CONTEXT_SIZE environment variable and restart the Bot Agent; larger "
                + "values use more RAM.",
                estimatedPromptTokens, maxTokens, budget, localCtx, localModelId, MAX_CONTEXT));
        }

        // cache_prompt was previously always true. Every call here is a fresh,
        // independent request (each bot action builds a complete standalone
        // prompt — none of them incrementally grow a running conversation), so
        // there was never a real prefix-reuse benefit to it. Worse: confirmed
        // via live testing that it actively corrupts output on some models —
        // gemma3-1b, asked to classify one text as "contract" right after an
        // unrelated prior request (a plain Q&A prompt, sharing no prefix) on
        // the same server, answered "receipt" — the same wrong category as its
        // immediately preceding call — with cache_prompt true, and answered
        // correctly with it false. The failure did not reproduce on Qwen3-4B
        // in the same scenario, but disabling it costs little (prompt
        // reprocessing for these short prompts is well under a second even on
        // the largest models here) against a real, silent correctness risk on
        // smaller models whenever a bot chains different actions against the
        // same loaded model.
        JsonObject body = new JsonObject();
        body.addProperty("prompt",       prompt);
        body.addProperty("n_predict",    maxTokens);
        body.addProperty("temperature",  temperature);
        body.addProperty("stream",       false);
        body.addProperty("cache_prompt", false);

        if (stopSequences != null && stopSequences.length > 0) {
            JsonArray stops = new JsonArray();
            for (String s : stopSequences) stops.add(s);
            body.add("stop", stops);
        }

        // Grammar-constrained generation: llama-server will reject any token that
        // would cause the output to deviate from the GBNF grammar at the current
        // position.  Only set when the caller explicitly requests it (e.g. JSON output).
        if (grammar != null && !grammar.isEmpty()) {
            body.addProperty("grammar", grammar);
        }

        // newBuilder() on an existing OkHttpClient shares the connection pool and
        // dispatcher — only timeout settings differ; no new thread pools are created.
        OkHttpClient timedClient = httpClient.newBuilder()
            .callTimeout(Duration.ofSeconds(timeoutSecs + 5))
            .readTimeout(Duration.ofSeconds(timeoutSecs + 5))
            .build();

        RequestBody reqBody = RequestBody.create(gson.toJson(body), JSON_TYPE);
        Request request = new Request.Builder()
            .url("http://" + SERVER_HOST + ":" + localPort + "/completion")
            .header("Authorization", "Bearer " + localApiKey)
            .post(reqBody)
            .build();

        try (Response response = timedClient.newCall(request).execute()) {
            // Read the body exactly once — OkHttp ResponseBody is single-use and
            // may be null (e.g. for responses with no body).
            ResponseBody responseBody = response.body();
            String bodyStr = responseBody != null ? responseBody.string() : "";

            if (!response.isSuccessful()) {
                throw new RuntimeException(
                    "llama-server /completion HTTP " + response.code()
                    + ": " + (bodyStr.isEmpty() ? "(no body)" : bodyStr));
            }

            JsonObject json = gson.fromJson(bodyStr, JsonObject.class);
            JsonElement contentEl = json != null ? json.get("content") : null;
            if (contentEl == null || contentEl.isJsonNull()) {
                throw new RuntimeException(
                    "llama-server response missing 'content' field. Full response: " + bodyStr);
            }
            return contentEl.getAsString();
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Stop / cleanup
    // ──────────────────────────────────────────────────────────────────────────

    public synchronized void stop() {
        stopInternal();
    }

    private void stopInternal() {
        if (serverProcess != null) {
            if (serverProcess.isAlive()) {
                logger.info("Stopping llama-server (model: {})", currentModelId);
                serverProcess.destroyForcibly();
                try { serverProcess.waitFor(); }
                catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            }
            serverProcess = null;
        }
        currentModelId        = null;
        apiKey                = null;
        port                  = -1;
        currentContextWindow  = -1;
    }

    public boolean isRunning() {
        Process proc = serverProcess; // single volatile read to avoid double-check races
        return proc != null && proc.isAlive();
    }

    public void shutdown() {
        stop();
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Utilities
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Finds a free local port by briefly binding to port 0.
     * setReuseAddress(true) helps the OS reclaim the port faster on some
     * platforms after the ServerSocket is closed.
     */
    /**
     * Rough token count estimate (~4 characters per token), used only for the
     * local pre-flight context-budget check in complete(). Not a real tokenizer
     * call — intentionally conservative via the 10% margin applied by the caller.
     */
    static int estimateTokenCount(String text) {
        if (text == null || text.isEmpty()) return 0;
        return (int) Math.ceil(text.length() / 4.0);
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket s = new ServerSocket(0)) {
            s.setReuseAddress(true);
            return s.getLocalPort();
        }
    }

    private static String readLogTail(Path logFile, int lines) {
        if (!Files.exists(logFile)) return "(log file not found)";
        try {
            List<String> all = Files.readAllLines(logFile);
            int from = Math.max(0, all.size() - lines);
            return String.join("\n", all.subList(from, all.size()));
        } catch (IOException e) {
            return "(could not read log: " + e.getMessage() + ")";
        }
    }
}
