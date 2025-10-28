import { OpenAIASRProvider } from "../../src/asr-providers/openai/provider";
import { TranscriptionRequest } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

describe("OpenAIASRProvider", () => {
  let provider: OpenAIASRProvider;

  beforeEach(() => {
    provider = new OpenAIASRProvider({
      apiKey: "test-api-key",
    });
  });

  describe("validateParams", () => {
    it("should not throw for supported parameters", () => {
      const params = {
        language: "en",
        prompt: "test prompt",
        response_format: "json",
        temperature: 0.5,
        timestamps: true,
        model: "whisper-1",
        file: Buffer.from("test")
      };

      expect(() => provider.validateParams(params)).not.toThrow();
    });

    it("should validate required parameters", () => {
      const params = {
        unsupported_param: "value",
        model: "whisper-1",
        file: Buffer.from("test")
      };

      expect(() => provider.validateParams(params)).not.toThrow();
    });
  });

  describe("translateParams", () => {
    it("should translate standard parameters correctly", () => {
      const params = {
        language: "en",
        prompt: "test prompt",
        response_format: "json",
        temperature: 0.5,
        timestamps: true,
        model: "whisper-1",
        file: Buffer.from("test")
      };

      const translated = provider.translateParams(params);
      expect(translated).toEqual({
        language: "en",
        prompt: "test prompt",
        response_format: "json",
        temperature: 0.5,
        timestamps: true,
      });
    });

    it("should retain other parameters", () => {
      const params = {
        custom_param: "value",
        model: "whisper-1",
        file: Buffer.from("test")
      };

      const translated = provider.translateParams(params);
      expect(translated).toEqual({
        custom_param: "value"
      });
    });
  });

  describe("transcribe", () => {
    it("should throw error if file is not provided", async () => {
      const request: TranscriptionRequest = {
        model: "openai:whisper-1",
        file: "",
      };

      await expect(provider.transcribe(request)).rejects.toThrow(AISuiteError);
    });
  });
});
