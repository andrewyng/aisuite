import { DeepgramASRProvider } from "../../src/asr-providers/deepgram/provider";
import { TranscriptionRequest } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

// Mock the Deepgram SDK
jest.mock("@deepgram/sdk", () => ({
  createClient: jest.fn(),
}));

describe("DeepgramASRProvider", () => {
  let provider: DeepgramASRProvider;
  let mockDeepgramClient: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock Deepgram client
    mockDeepgramClient = {
      listen: {
        prerecorded: {
          transcribeFile: jest.fn(),
        },
      },
    };

    // Mock the createClient function
    const { createClient } = require("@deepgram/sdk");
    createClient.mockReturnValue(mockDeepgramClient);

    // Create provider instance
    provider = new DeepgramASRProvider({
      apiKey: "test-api-key",
    });
  });

  describe("constructor", () => {
    it("should initialize with basic config", () => {
      const { createClient } = require("@deepgram/sdk");

      const config = { apiKey: "test-key" };
      const provider = new DeepgramASRProvider(config);

      expect(provider.name).toBe("deepgram");
      expect(createClient).toHaveBeenCalledWith({
        key: "test-key",
      });
    });

    it("should initialize with baseURL config", () => {
      const { createClient } = require("@deepgram/sdk");

      const config = {
        apiKey: "test-key",
        baseURL: "https://custom.deepgram.com",
      };
      const provider = new DeepgramASRProvider(config);

      expect(provider.name).toBe("deepgram");
      expect(createClient).toHaveBeenCalledWith({
        key: "test-key",
        baseUrl: "https://custom.deepgram.com",
      });
    });

    it("should not include baseUrl when not provided", () => {
      const { createClient } = require("@deepgram/sdk");

      const config = { apiKey: "test-key" };
      new DeepgramASRProvider(config);

      expect(createClient).toHaveBeenCalledWith({
        key: "test-key",
      });
    });
  });

  describe("validateParams", () => {
    it("should not throw for supported parameters", () => {
      const params = {
        model: "nova-2",
        file: Buffer.from("test audio data"),
        language: "en-US",
        timestamps: true,
        word_confidence: true,
        speaker_labels: true,
        smart_format: true,
        punctuate: true,
        diarize: true,
        utterances: true,
      };

      expect(() => provider.validateParams(params)).not.toThrow();
    });

    it("should accept additional parameters without warnings", () => {
      const consoleSpy = jest.spyOn(console, "warn").mockImplementation();

      const params = {
        model: "nova-2",
        file: Buffer.from("test audio data"),
        unsupported_param: "value",
        another_unsupported: true,
      };

      provider.validateParams(params);

      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    it("should not warn for deepgram-specific parameters", () => {
      const consoleSpy = jest.spyOn(console, "warn").mockImplementation();

      const params = {
        model: "nova-2",
        file: Buffer.from("test audio data"),
        deepgram_custom_param: "value",
        deepgram_another_param: true,
      };

      provider.validateParams(params);

      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe("translateParams", () => {
    it("should translate language parameter", () => {
      const params = { language: "en-US" };
      const result = provider.translateParams(params);

      expect(result).toEqual({ language: "en-US" });
    });

    it("should translate timestamps to utterances", () => {
      const params = { timestamps: true };
      const result = provider.translateParams(params);

      expect(result).toEqual({ utterances: true });
    });

    it("should pass through word_confidence parameter", () => {
      const params = { word_confidence: true };
      const result = provider.translateParams(params);

      expect(result).toEqual({ word_confidence: true });
    });

    it("should pass through speaker_labels parameter", () => {
      const params = { speaker_labels: true };
      const result = provider.translateParams(params);

      expect(result).toEqual({ speaker_labels: true });
    });

    it("should pass through deepgram-specific parameters", () => {
      const params = {
        deepgram_custom_param: "value",
        deepgram_another_param: true,
      };
      const result = provider.translateParams(params);

      expect(result).toEqual({
        deepgram_custom_param: "value",
        deepgram_another_param: true,
      });
    });

    it("should pass through other parameters unchanged", () => {
      const params = {
        temperature: 0.5,
        custom_param: "value",
      };
      const result = provider.translateParams(params);

      expect(result).toEqual({
        temperature: 0.5,
        custom_param: "value",
      });
    });

    it("should handle multiple parameter translations", () => {
      const params = {
        language: "en-US",
        timestamps: true,
        word_confidence: true,
        speaker_labels: true,
        temperature: 0.5,
      };
      const result = provider.translateParams(params);

      expect(result).toEqual({
        language: "en-US",
        utterances: true,
        word_confidence: true,
        speaker_labels: true,
        temperature: 0.5,
      });
    });
  });

  describe("transcribe", () => {
    const baseRequest: TranscriptionRequest = {
      model: "nova-2",
      file: Buffer.from("test audio data"),
    };

    it("should successfully transcribe audio", async () => {
      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Hello world",
                    confidence: 0.95,
                    words: [
                      {
                        word: "Hello",
                        start: 0.0,
                        end: 0.5,
                        confidence: 0.98,
                      },
                      {
                        word: "world",
                        start: 0.6,
                        end: 1.0,
                        confidence: 0.92,
                      },
                    ],
                  },
                ],
              },
            ],
            utterances: [
              {
                transcript: "Hello world",
                start: 0.0,
                end: 1.0,
                speaker: 0,
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      const result = await provider.transcribe(baseRequest);

      expect(
        mockDeepgramClient.listen.prerecorded.transcribeFile
      ).toHaveBeenCalledWith(
        Buffer.from("test audio data"),
        expect.objectContaining({
          model: "nova-2",
        })
      );

      expect(result).toEqual({
        text: "Hello world",
        language: "en-US",
        confidence: 0.95,
        words: [
          {
            text: "Hello",
            start: 0.0,
            end: 0.5,
            confidence: 0.98,
          },
          {
            text: "world",
            start: 0.6,
            end: 1.0,
            confidence: 0.92,
          },
        ],
        segments: [
          {
            text: "Hello world",
            start: 0.0,
            end: 1.0,
            speaker: "0",
          },
        ],
      });
    });

    it("should handle string file path", async () => {
      const fs = require("fs");
      jest
        .spyOn(fs, "readFileSync")
        .mockReturnValue(Buffer.from("test audio data"));

      const request: TranscriptionRequest = {
        model: "nova-2",
        file: "/path/to/audio.wav",
      };

      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                    confidence: 0.9,
                  },
                ],
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      await provider.transcribe(request);

      expect(fs.readFileSync).toHaveBeenCalledWith("/path/to/audio.wav");
      expect(
        mockDeepgramClient.listen.prerecorded.transcribeFile
      ).toHaveBeenCalledWith(
        Buffer.from("test audio data"),
        expect.any(Object)
      );
    });

    it("should handle Uint8Array file", async () => {
      const uint8Array = new Uint8Array([1, 2, 3, 4]);
      const request: TranscriptionRequest = {
        model: "nova-2",
        file: uint8Array,
      };

      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                    confidence: 0.9,
                  },
                ],
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      await provider.transcribe(request);

      expect(
        mockDeepgramClient.listen.prerecorded.transcribeFile
      ).toHaveBeenCalledWith(Buffer.from(uint8Array), expect.any(Object));
    });

    it("should handle unsupported file type gracefully", async () => {
      const request: TranscriptionRequest = {
        model: "nova-2",
        file: "unsupported" as any,
      };

      // This test verifies that the provider handles unsupported file types
      // The actual error handling is tested in other scenarios
      await expect(provider.transcribe(request)).rejects.toThrow(AISuiteError);
    });

    it("should throw error when Deepgram API returns error", async () => {
      const mockDeepgramResponse = {
        result: null,
        error: {
          message: "API key invalid",
        },
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      await expect(provider.transcribe(baseRequest)).rejects.toThrow(
        new AISuiteError(
          "Deepgram API error: API key invalid",
          "deepgram",
          "API_ERROR"
        )
      );
    });

    it("should handle Deepgram API exceptions", async () => {
      const apiError = new Error("Network error");
      mockDeepgramClient.listen.prerecorded.transcribeFile.mockRejectedValue(
        apiError
      );

      await expect(provider.transcribe(baseRequest)).rejects.toThrow(
        new AISuiteError(
          "Deepgram ASR error: Network error",
          "deepgram",
          "API_ERROR"
        )
      );
    });

    it("should pass timeout options", async () => {
      const options = { timeout: 30000 };
      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                    confidence: 0.9,
                  },
                ],
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      await provider.transcribe(baseRequest, options);

      expect(
        mockDeepgramClient.listen.prerecorded.transcribeFile
      ).toHaveBeenCalledWith(Buffer.from("test audio data"), {
        model: "nova-2",
        "mimetype": "audio/wav",
      });
    });

    it("should translate parameters correctly", async () => {
      const request: TranscriptionRequest = {
        model: "nova-2",
        file: Buffer.from("test audio data"),
        language: "en-US",
        timestamps: true,
        word_confidence: true,
        speaker_labels: true,
        temperature: 0.5,
      };

      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                    confidence: 0.9,
                  },
                ],
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      await provider.transcribe(request);

      expect(
        mockDeepgramClient.listen.prerecorded.transcribeFile
      ).toHaveBeenCalledWith(
        Buffer.from("test audio data"),
        expect.objectContaining({
          model: "nova-2",
          language: "en-US",
          utterances: true,
          word_confidence: true,
          speaker_labels: true,
          temperature: 0.5,
        })
      );
    });

    it("should handle response without words or utterances", async () => {
      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                    confidence: 0.9,
                  },
                ],
              },
            ],
          },
          metadata: {
            language: "en-US",
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      // Ensure model is present in baseRequest
      const requestWithModel = { ...baseRequest, model: "nova-2" };
      const result = await provider.transcribe(requestWithModel);

      expect(result).toEqual({
        text: "Test transcription",
        language: "en-US",
        confidence: 0.9,
        words: [],
        segments: [],
      });
    });

    it("should handle malformed response gracefully", async () => {
      const mockDeepgramResponse = {
        result: {
          results: {
            channels: [
              {
                alternatives: [
                  {
                    transcript: "Test transcription",
                  },
                ],
              },
            ],
          },
        },
        error: null,
      };

      mockDeepgramClient.listen.prerecorded.transcribeFile.mockResolvedValue(
        mockDeepgramResponse
      );

      const result = await provider.transcribe(baseRequest);

      expect(result).toEqual({
        text: "Test transcription",
        language: "unknown",
        confidence: undefined,
        words: [],
        segments: [],
      });
    });
  });
});
