# LocalAI - On-Device Small Language Model Package for Automation Anywhere

![Java 11+](https://img.shields.io/badge/Java-11%2B-informational)
![Automation Anywhere](https://img.shields.io/badge/Automation%20Anywhere-A360-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9481-green)
![Version](https://img.shields.io/badge/version-2.12.67-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey)

Run AI inference **entirely on-device** with no cloud APIs, no GPU, and no internet required at runtime (after initial setup). This Automation Anywhere package brings Small Language Models (SLMs) directly into your bot workflows using the official [llama.cpp](https://github.com/ggerganov/llama.cpp) CPU binaries and quantized GGUF models.

## Why On-Device AI?

- **Data stays local** - Sensitive data never leaves the machine. No API keys, no cloud endpoints.
- **No GPU required** - Runs on standard CPUs (Intel x64, Apple Silicon ARM64).
- **Zero runtime cost** - No per-token billing or API rate limits.
- **Offline capable** - After initial model download and binary setup, works completely offline.
- **Cross-platform** - Windows x64 and macOS (Apple Silicon ARM64 + Intel x64).

## Supported Models

| Model | Parameters | Size | Context | Best For |
|-------|-----------|------|---------|----------|
| `qwen3-4b` | 4B | ~2.5GB | 32K | Best for structured output, JSON (default) |
| `gemma3-4b` | 4B | ~2.5GB | 128K | Balanced all-rounder |
| `phi4-mini` | 3.8B | ~2.5GB | 128K | Instructions & long-context reasoning |
| `gemma4-e2b` | 5.1B | ~3.1GB | 128K | Highest quality output |
| `llama3.2-3b` | 3B | ~2.0GB | 8K | Fast, proven baseline |
| `deepseek-r1-1.5b` | 1.5B | ~1.1GB | 128K | Chain-of-thought reasoning |
| `qwen2.5-coder-3b` | 3B | ~1.9GB | 32K | Code generation & scripting |
| `gemma3-270m` | 270M | ~253MB | 32K | Smallest & fastest; verified reliable only for the Prompt action's general Q&A. Not offered in Classify Text, Redact PII, Transform to JSON, or Sanitize JSON — tested and found to fabricate or drop data (see model dropdowns for details) |
| `gemma3-1b` | 1B | ~769MB | 32K | Small & fast (~2.8x faster generation than qwen3-4b); verified reliable for Prompt, Classify Text, Redact PII, Transform to JSON, Sanitize JSON, and Summarize Text. Not offered in Extract Data or Normalize & Standardize — tested and found unreliable for those specific tasks (see model dropdowns for details) |

Models are automatically downloaded from HuggingFace on first use and cached at:
- **Windows:** `%LOCALAPPDATA%\AutomationAnywhere\LocalAI\`
- **macOS:** `~/localAI/`

## How It Works

This package uses the official llama.cpp CPU release binaries rather than a JNI binding. On first run, `ValidateDevice` downloads the appropriate llama-server binary (~15MB) for your platform. Each inference action starts `llama-server` as a subprocess on a free local port and communicates with it over HTTP — the same architecture llama.cpp uses for its own web interface.

This approach means:
- **All models work** — no architecture limitations from outdated JNI bindings
- **Automatic updates** — new model families (Qwen3, Gemma 4, etc.) are supported immediately
- **Identical behavior on Windows and macOS** — same binary, same HTTP API

### Network overrides

By default the inference server binds to `127.0.0.1` on a random free port — zero configuration needed for the overwhelming majority of installs. Two rare situations need an override:

- **Loopback doesn't route to the bot's own child process** (some VM/VDI network virtualization or VPN split-tunnel software): set the `LOCALAI_SERVER_HOST` environment variable to a working address (or `-Dlocalai.server.host=<ip>` as a JVM argument).
- **Firewall/network policy only allows a specific, known port** (a random port every run can't be allowlisted): every inference action has a **Server Port (Advanced)** field — set it to your allowed port instead of editing environment variables per machine. Leave it at `0` for normal automatic behavior. A `LOCALAI_SERVER_PORT` environment variable (or `-Dlocalai.server.port=<port>`) is also available as a machine-wide fallback if you'd rather not set it per action, but the action field takes precedence when both are set.

## Actions

### Validate Device

**Run this once before using any inference actions.** Downloads and validates the llama-server binary for your platform, and optionally pre-downloads a model.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| Model | Select | Which model to download/validate |

**Returns:** Boolean - `true` if the device is ready for inference.

---

### Prompt

Send any text prompt to a local SLM and get a response. Use for question answering, summarization, extraction, rewriting, or any general-purpose text task.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Prompt Text | Text Area | *required* | Your instruction or question |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |
| Temperature | Number | 0.3 | Randomness (0.0 = deterministic, 1.0 = creative) |

**Returns:** Dictionary with key `response` (String).

---

### Classify Text

Categorize text into one of your predefined categories. Useful for routing tickets, tagging content, or sentiment analysis.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Text to classify |
| Categories | Text | *required* | Comma-separated list (e.g., `positive, negative, neutral`) |
| Model | Select | `qwen3-4b` | Which model to use |
| Include Confidence | Checkbox | false | Append confidence score (0-1) |
| Include Explanation | Checkbox | false | Append brief explanation |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — `category` or `category|0.95|explanation` when options enabled.

---

### Transform to JSON

Convert CSV, TSV, key-value pairs, tables, or lists into valid JSON.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Data to convert |
| Input Format | Select | `auto-detect` | csv, tsv, key-value, table, list, auto-detect |
| Output Style | Select | `compact` | compact or pretty |
| Output Type | Select | `array` | object `{}` or array `[{}]` |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — validated JSON output.

---

### Normalize and Standardize

Clean up inconsistent data formats — dates, phone numbers, addresses, names, emails.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Data to normalize |
| Data Type | Select | `auto-detect` | phone, date, address, name, email, auto-detect, custom |
| Output Format | Text | *(varies)* | Target format (e.g., `YYYY-MM-DD`) |
| Model | Select | `qwen3-4b` | Which model to use |
| Preserve Original on Failure | Checkbox | true | Return input if normalization fails |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — normalized/standardized text.

---

### Sanitize JSON

Make arbitrary text safe for embedding in JSON strings. Uses model-based intelligent sanitization with a rule-based fallback.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text | *required* | Text to sanitize |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — JSON-safe text with properly escaped characters.

---

### Extract Data

Extract structured fields from unstructured text (invoices, emails, forms, etc.).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Text to extract from |
| Fields to Extract | Text | *required* | Comma-separated field names |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** Dictionary of extracted field → value pairs.

---

### Summarize Text

Condense long text into a concise summary.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Text to summarize |
| Summary Length | Select | `medium` | short / medium / long |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — the summary.

---

### Redact PII

Identify and redact personally identifiable information from text.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Input Text | Text Area | *required* | Text to redact |
| PII Types | Text | *required* | Comma-separated types (e.g., `name, email, phone, ssn`) |
| Replacement | Text | `[REDACTED]` | Replacement token |
| Model | Select | `qwen3-4b` | Which model to use |
| Timeout (seconds) | Number | 30 | Max generation time |
| Server Port (Advanced) | Number | 0 | Pin to a specific TCP port instead of a random one; only needed in locked-down network/firewall setups. 0 = automatic |

**Returns:** String — text with PII replaced.

---

### Delete Model

Remove a downloaded model from disk to free space.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| Model | Select | Which model to delete |

**Returns:** Boolean - `true` if the model was deleted (or didn't exist).

---

## Building from Source

### Prerequisites

- Java 11+
- Gradle (wrapper included)
- Internet access during build (to download dependencies)

### Build the package

```shell
./gradlew clean build
```

The output JAR is at `build/libs/LocalAI-2.12.{buildNum}.jar`.

### Run tests

```shell
./gradlew test
```

> Tests that perform inference require the model to be downloaded (~2.5GB for Qwen3 4B) and the llama-server binary to be installed. Run `ValidateDevice` in a bot first, or the inference tests will trigger the download automatically.

## Architecture

```
src/main/java/com/automationanywhere/botcommand/
├── Prompt.java                  # General text generation action
├── ClassifyText.java            # Text classification action
├── TransformToJSON.java         # Format-to-JSON conversion action
├── NormalizeAndStandardize.java # Data normalization action
├── SanitizeJSON.java            # JSON text sanitization action
├── ExtractData.java             # Field extraction action
├── SummarizeText.java           # Text summarization action
├── RedactPII.java               # PII redaction action
├── ValidateDevice.java          # Binary + model download/validation action
├── DeleteModel.java             # Model cleanup action
└── utils/
    ├── ModelManager.java        # Model registry, paths, and cache directory
    ├── LlamaBinaryManager.java  # Downloads and manages the llama-server binary
    ├── LlamaServerManager.java  # Manages the llama-server subprocess lifecycle
    ├── LlamaInference.java      # Chat template formatting and inference entry point
    └── ModelDownloader.java     # HuggingFace model downloader
```

### Inference Flow

```
Bot Action (ClassifyText, Prompt, etc.)
    └── LlamaInference
            ├── applyChatTemplate()    — formats prompt per model family
            └── LlamaServerManager
                    ├── ensureModelLoaded()  — starts/reuses llama-server subprocess
                    └── complete()           — HTTP POST to /completion
```

**LlamaBinaryManager** fetches the official llama.cpp CPU release (currently b9481) from GitHub on first use and caches it to disk. Windows gets `llama-server.exe` from the zip; macOS gets `llama-server` from the tar.gz. The binary is extracted to `{modelCacheDir}/bin/`.

**LlamaServerManager** is a singleton that keeps `llama-server` running between inference calls. When a different model is requested it stops the current server and starts a new one. The server binds to a random free port on `127.0.0.1`.

**LlamaInference** applies the correct chat template for each model family before sending to the server: `CHATML_QWEN3` for Qwen3, `CHATML` for Qwen2.5/DeepSeek, `GEMMA3` for Gemma 3, `GEMMA4` for Gemma 4, `PHI4` for Phi-4 Mini.

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | b9481 (subprocess) | Local LLM inference — downloaded at runtime, MIT licensed |
| [OkHttp](https://square.github.io/okhttp/) | 4.12.0 | Model downloads from HuggingFace + HTTP calls to llama-server |
| [Gson](https://github.com/google/gson) | 2.10.1 | JSON validation and formatting |
| Automation Anywhere SDK | 1.6.0 | Package SDK for A360 |

## Windows Notes

- llama-server binary and models are stored under `%LOCALAPPDATA%\AutomationAnywhere\LocalAI\` to avoid Windows Controlled Folder Access (CFA) restrictions.
- `--no-mmap` is passed automatically on Windows to prevent memory-mapped file issues.
- The llama-server process sets its working directory to the `bin\` folder so that Windows can locate its DLL dependencies.

## License

Proprietary - Automation Anywhere

llama.cpp is used under the [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE).
