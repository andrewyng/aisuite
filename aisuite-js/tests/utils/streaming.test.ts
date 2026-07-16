import { createChunk, generateId } from "../../src/utils/streaming";

describe("Streaming Utils", () => {
  describe("createChunk", () => {
    it("should create a basic chunk with required fields", () => {
      const id = "test-chunk-id";
      const model = "gpt-4";

      const chunk = createChunk(id, model);

      expect(chunk).toEqual({
        id,
        object: "chat.completion.chunk",
        created: expect.any(Number),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: undefined,
              tool_calls: undefined,
            },
            finish_reason: undefined,
          },
        ],
      });
    });

    it("should create a chunk with content", () => {
      const id = "test-chunk-id";
      const model = "claude-3-sonnet";
      const content = "Hello, world!";

      const chunk = createChunk(id, model, content);

      expect(chunk).toEqual({
        id,
        object: "chat.completion.chunk",
        created: expect.any(Number),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content,
              tool_calls: undefined,
            },
            finish_reason: undefined,
          },
        ],
      });
    });

    it("should create a chunk with finish reason", () => {
      const id = "test-chunk-id";
      const model = "mistral-large";
      const finishReason = "stop";

      const chunk = createChunk(id, model, undefined, finishReason);

      expect(chunk).toEqual({
        id,
        object: "chat.completion.chunk",
        created: expect.any(Number),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: undefined,
              tool_calls: undefined,
            },
            finish_reason: finishReason,
          },
        ],
      });
    });

    it("should create a chunk with tool calls", () => {
      const id = "test-chunk-id";
      const model = "gpt-4";
      const toolCalls = [
        {
          id: "call-1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"location": "New York"}',
          },
        },
      ];

      const chunk = createChunk(id, model, undefined, undefined, toolCalls);

      expect(chunk).toEqual({
        id,
        object: "chat.completion.chunk",
        created: expect.any(Number),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: undefined,
              tool_calls: toolCalls,
            },
            finish_reason: undefined,
          },
        ],
      });
    });

    it("should create a complete chunk with all parameters", () => {
      const id = "test-chunk-id";
      const model = "gpt-4";
      const content = "The weather is sunny";
      const finishReason = "stop";
      const toolCalls = [
        {
          id: "call-1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"location": "New York"}',
          },
        },
      ];

      const chunk = createChunk(id, model, content, finishReason, toolCalls);

      expect(chunk).toEqual({
        id,
        object: "chat.completion.chunk",
        created: expect.any(Number),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content,
              tool_calls: toolCalls,
            },
            finish_reason: finishReason,
          },
        ],
      });
    });

    it("should set created timestamp to current time", () => {
      const before = Math.floor(Date.now() / 1000);
      const chunk = createChunk("test-id", "test-model");
      const after = Math.floor(Date.now() / 1000);

      expect(chunk.created).toBeGreaterThanOrEqual(before);
      expect(chunk.created).toBeLessThanOrEqual(after);
    });

    it("should always set index to 0", () => {
      const chunk = createChunk("test-id", "test-model");

      expect(chunk.choices[0].index).toBe(0);
    });

    it("should always set role to assistant", () => {
      const chunk = createChunk("test-id", "test-model");

      expect(chunk.choices[0].delta.role).toBe("assistant");
    });
  });

  describe("generateId", () => {
    it("should generate a string id", () => {
      const id = generateId();

      expect(typeof id).toBe("string");
      expect(id.length).toBeGreaterThan(0);
    });

    it("should generate ids with chatcmpl prefix", () => {
      const id = generateId();

      expect(id).toMatch(/^chatcmpl-/);
    });

    it("should generate unique ids", () => {
      const id1 = generateId();
      const id2 = generateId();
      const id3 = generateId();

      expect(id1).not.toBe(id2);
      expect(id1).not.toBe(id3);
      expect(id2).not.toBe(id3);
    });

    it("should generate ids with consistent format", () => {
      const id = generateId();

      // Should match pattern: chatcmpl- followed by 9 alphanumeric characters
      expect(id).toMatch(/^chatcmpl-[a-z0-9]{9}$/);
    });

    it("should generate multiple ids without conflicts", () => {
      const ids = new Set();
      const iterations = 1000;

      for (let i = 0; i < iterations; i++) {
        ids.add(generateId());
      }

      // All ids should be unique
      expect(ids.size).toBe(iterations);
    });
  });

  describe("integration", () => {
    it("should create chunks with generated ids", () => {
      const model = "test-model";
      const content = "test content";

      const chunk = createChunk(generateId(), model, content);

      expect(chunk.id).toMatch(/^chatcmpl-/);
      expect(chunk.model).toBe(model);
      expect(chunk.choices[0].delta.content).toBe(content);
    });

    it("should create multiple chunks with different ids", () => {
      const model = "test-model";

      const chunk1 = createChunk(generateId(), model);
      const chunk2 = createChunk(generateId(), model);

      expect(chunk1.id).not.toBe(chunk2.id);
      expect(chunk1.created).toBe(chunk2.created); // Should be created at same time
    });
  });
});
