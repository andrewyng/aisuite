# ASR - API Parameter Design Philosophy

## Design Goal: Portable Code with Provider Flexibility

The ASR parameter system is designed around a core principle: **developers should write portable code that works across providers, while retaining the ability to use provider-specific features when needed**. This document explains the rationale behind our parameter classification and validation approach.

---

## Mandatory Parameters and Common Mappings

### The Foundation: Minimal Requirements

Every transcription needs just two things:
- **`model`**: Which model/provider to use
- **`file`**: What audio to transcribe

By keeping mandatory parameters minimal, we maximize compatibility and reduce the barrier to getting started.

### Common Parameters: Write Once, Run Anywhere

Beyond the basics, there are concepts that exist across providers but use different names or formats. We handle three common parameters that auto-map to each provider's native API:

**Example: Same code, different providers**

```python
# Works with OpenAI
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="meeting.mp3",
    language="en",
    prompt="discussion about API design"
)

# Exact same code works with Deepgram
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3",
    language="en",
    prompt="discussion about API design"
)
```

Behind the scenes:
- **`language`** passes through as `language` for both OpenAI and Deepgram, but expands to `language_code: "en-US"` for Google
- **`prompt`** passes as `prompt` to OpenAI, transforms to `keywords: ["discussion", "about", "API", "design"]` for Deepgram, and becomes `speech_contexts: [{"phrases": ["discussion about API design"]}]` for Google
- **`temperature`** passes through to OpenAI (which supports it) and is silently ignored by Deepgram and Google (which don't)

**Why auto-mapping?** Developers shouldn't need to remember that Google uses `language_code` while others use `language`, or that Deepgram expects a list of keywords. The framework handles these provider quirks transparently, letting you write portable code.

---

## Provider-Specific Features: Pass-Through for Power Users

Each provider has unique features that give them competitive advantages. We don't limit you to the "lowest common denominator" - if you need provider-specific functionality, it's available:

**Deepgram's advanced features:**
```python
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3",
    language="en",
    punctuate=True,        # Deepgram-specific
    diarize=True,          # Deepgram-specific
    sentiment=True,        # Deepgram-specific
    smart_format=True      # Deepgram-specific
)
```

**Google's speech contexts:**
```python
result = client.audio.transcriptions.create(
    model="google:latest_long",
    file="meeting.mp3",
    language_code="en-US",
    enable_automatic_punctuation=True,  # Google-specific
    max_alternatives=3,                  # Google-specific
    speech_contexts=[{"phrases": ["API", "SDK", "REST"]}]  # Google-specific
)
```

These provider-specific parameters pass through directly to the provider's SDK. The framework validates them based on your configured mode (see next section), but doesn't block access to unique features.

---

## Progressive Validation: Safety When You Need It

The validation system supports three modes to match different development stages:

### Development Mode: `"warn"` (Default)
```python
client = Client(extra_param_mode="warn")
```
Unknown parameters trigger warnings but continue execution. Perfect for exploration and prototyping. You see *"OpenAI doesn't support 'punctuate'"* but your code keeps running.

### Strict Mode: `"strict"`
```python
client = Client(extra_param_mode="strict")
```
Unknown parameters raise errors immediately. Use in production to catch typos, configuration mistakes, or provider API changes early. Ensures no silent failures.

### Permissive Mode: `"permissive"`
```python
client = Client(extra_param_mode="permissive")
```
All parameters pass through without validation. Use for beta features, experimental parameters, or when providers add new capabilities faster than framework updates.

**Progressive workflow:**
1. **Develop** with `warn` - explore freely, see warnings
2. **Refactor** - fix warnings to make code portable
3. **Deploy** with `strict` - ensure production safety

---

## Developer Experience Benefits

### 1. Write Portable Code Naturally
The same parameter names work across providers. Switch from OpenAI to Deepgram by changing one word: the model identifier.

### 2. Progressive Enhancement
Start with portable common parameters. Add provider-specific features only where you need them. Your core logic remains portable even when using advanced features for specific providers.

### 3. Zero Framework Lock-in
Parameter names come directly from provider APIs, not framework abstractions. If you need to remove the framework, you already know the native API - the names are identical.

### 4. Validation That Adapts to You
Choose your safety level based on context. Strict for production, warn for development, permissive for bleeding-edge features. The framework supports your workflow rather than constraining it.

### 5. No Documentation Friction
Copy parameters from provider docs directly. No need to learn our abstraction layer or figure out mappings - we handle the common cases, you use native names for everything else.

---

## Alternative Design Considered

We considered creating a unified options object (`TranscriptionOptions`) that explicitly defines all parameters with framework-specific names. We chose pass-through instead because:

1. **Provider APIs evolve faster than frameworks** - New parameters appear frequently. Pass-through lets developers use them immediately (in permissive mode) without waiting for framework updates.

2. **Provider features don't map cleanly** - Deepgram's sentiment analysis, Google's complex speech contexts, OpenAI's timestamp granularities - each is unique. A unified object means either losing functionality or creating complex provider-specific abstractions.

3. **Direct API access reduces friction** - Developers already know their provider's API from official docs. They can use parameter names directly rather than learning another abstraction layer.

The pass-through approach with progressive validation provides the best of both worlds: portability for common cases, power for advanced features, and safety when you need it.

---

## Design Principles Summary

- **Mandatory Minimal**: Only `model` and `file` required
- **Common Auto-Mapped**: Frequent cross-provider concepts map transparently
- **Provider-Specific Pass-Through**: Unique features remain accessible
- **Progressive Validation**: Three modes for different development stages
- **Zero Abstraction Tax**: Use provider APIs directly with optional safety nets

This design prioritizes developer experience through portability without sacrificing power, validation without blocking experimentation, and simplicity without limiting functionality.
