import { Client } from "../src";
import * as fs from "fs";
import * as path from "path";

async function main() {
  // Initialize the client with OpenAI configuration
  const client = new Client({
    openai: {
      apiKey: process.env.OPENAI_API_KEY || "your-openai-api-key",
      audio: true,
    },
  });

  console.log("Available ASR providers:", client.listASRProviders());

  // Example: Transcribe an audio file
  try {
    // Path to your audio file
    const testAudioPath = path.join("test-audio.wav");

    // Check if test file exists
    if (!fs.existsSync(testAudioPath)) {
      console.log(
        "Test audio file not found. Please provide a valid audio file for transcription."
      );
      console.log("Expected path:", testAudioPath);
      return;
    }

    const audioBuffer = fs.readFileSync(testAudioPath);

    // Transcribe using OpenAI Whisper model
    const result = await client.audio.transcriptions.create({
      model: "openai:whisper-1",
      file: audioBuffer,
      language: "en",
      response_format: "verbose_json",
      temperature: 0,
      timestamps: true,
    });

    console.log("Transcription Result:");
    console.log("Text:", result.text);
    console.log("Language:", result.language);
    console.log("Confidence:", result.confidence);

    if (result.words && result.words.length > 0) {
      console.log("\nWords with timestamps:");
      result.words.slice(0, 5).forEach((word, index) => {
        console.log(
          `${index + 1}. "${word.text}" (${word.start}s - ${word.end}s, confidence: ${word.confidence})`
        );
      });
    }

    if (result.segments && result.segments.length > 0) {
      console.log("\nSegments:");
      result.segments.slice(0, 3).forEach((segment, index) => {
        console.log(
          `${index + 1}. "${segment.text}" (${segment.start}s - ${segment.end}s)`
        );
      });
    }
  } catch (error) {
    console.error("Error:", error);
  }
}

main().catch(console.error);
