import { Client } from "../src";
import * as fs from "fs";
import * as path from "path";

async function main() {
  // Initialize the client with Deepgram configuration
  // Using Deepgram SDK v4.11.2 with the new createClient API
  const client = new Client({
    deepgram: {
      apiKey: process.env.DEEPGRAM_API_KEY || "your-deepgram-api-key",
    },
  });

  console.log("Available ASR providers:", client.listASRProviders());

  // Example: Transcribe an audio file
  try {
    // Create a simple test audio file (you would replace this with your actual audio file)
    const testAudioPath = path.join("test-audio.wav");

    // Check if test file exists, if not create a placeholder
    if (!fs.existsSync(testAudioPath)) {
      console.log(
        "Test audio file not found. Please provide a valid audio file for transcription."
      );
      console.log("Expected path:", testAudioPath);
      return;
    }

    // Read the file as a buffer
    const audioBuffer = fs.readFileSync(testAudioPath);

    // Create the transcription request with the audio buffer
    const result = await client.audio.transcriptions.create({
      model: "deepgram:general",
      file: audioBuffer,
      language: "en-US",
      timestamps: true,
      word_confidence: true,
      speaker_labels: true,
    });

    console.log("Transcription Result:");
    console.log("Text:", result.text);
    console.log("Language:", result.language);
    console.log("Confidence:", result.confidence);

    if (result.words && result.words.length > 0) {
      console.log("\nWords with timestamps:");
      result.words.slice(0, 5).forEach((word, index) => {
        console.log(
          `${index + 1}. "${word.text}" (${word.start}s - ${
            word.end
          }s, confidence: ${word.confidence})`
        );
      });
    }

    if (result.segments && result.segments.length > 0) {
      console.log("\nSegments:");
      result.segments.forEach((segment, index) => {
        console.log(
          `${index + 1}. [${segment.start}s - ${segment.end}s] ${segment.text}`
        );
      });
    }
  } catch (error) {
    console.error("Error during transcription:", error);
  }
}

main().catch(console.error);

