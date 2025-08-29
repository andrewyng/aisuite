import { OpenAIASRProvider } from "../../src/providers/openai/audio";
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
      };

      expect(() => provider.validateParams("whisper-1", params)).not.toThrow();
    });

    it("should log warning for unsupported parameters", () => {
      const consoleSpy = jest.spyOn(console, "warn");
      const params = {
        unsupported_param: "value",
      };

      provider.validateParams("whisper-1", params);
      expect(consoleSpy).toHaveBeenCalledWith(
        "Parameter 'unsupported_param' may not be supported by OpenAI ASR"
      );
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
      };

      const translated = provider.translateParams("whisper-1", params);
      expect(translated).toEqual({
        language: "en",
        prompt: "test prompt",
        response_format: "json",
        temperature: 0.5,
        timestamp_granularities: ["word"],
      });
    });

    it("should include OpenAI-specific parameters", () => {
      const params = {
        openai_custom_param: "value",
      };

      const translated = provider.translateParams("whisper-1", params);
      expect(translated).toEqual({
        custom_param: "value",
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
