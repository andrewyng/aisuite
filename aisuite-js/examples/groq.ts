import "dotenv/config";
import { Client, ChatCompletionResponse, ChatMessage } from "../src";

// Available Groq models
const AVAILABLE_MODELS = {
  MIXTRAL: "groq:mistral-saba-24b",
  LLAMA2: "groq:llama-3.3-70b-versatile",
  GEMMA: "groq:gemma2-9b-it",
};

async function main() {
  const client = new Client({
    groq: { apiKey: process.env.GROQ_API_KEY! },
  });

  console.log("\n🚀 Groq Chat Examples\n");

  // Example 1: Basic chat completion with Mixtral
  console.log("--- Basic Chat Completion with Mixtral ---");
  try {
    const response = (await client.chat.completions.create({
      model: AVAILABLE_MODELS.MIXTRAL,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is TypeScript in one sentence?" },
      ],
      temperature: 0.7,
      max_tokens: 100,
      stream: false,
    })) as ChatCompletionResponse;

    console.log("Response:", response.choices[0].message.content);
    console.log("Usage:", response.usage);
    console.log("Full response:", JSON.stringify(response, null, 2));
  } catch (error) {
    console.error("Error:", error);
  }

  // Example 2: Streaming with LLaMA2
  console.log("\n--- Streaming Example with LLaMA2 ---");
  try {
    const stream = await client.chat.completions.create({
      model: AVAILABLE_MODELS.LLAMA2,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        {
          role: "user",
          content: "Write a haiku about artificial intelligence.",
        },
      ],
      stream: true,
      temperature: 0.7,
      max_tokens: 100,
    });

    console.log("Response:");
    let fullContent = "";
    for await (const chunk of stream as AsyncIterable<any>) {
      const content = chunk.choices[0]?.delta?.content || "";
      process.stdout.write(content);
      fullContent += content;
    }
    console.log("\n");
  } catch (error) {
    console.error("Streaming error:", error);
  }

  // Example 3: Chat completion with Gemma
  console.log("\n--- Chat Completion with Gemma ---");
  try {
    const response = (await client.chat.completions.create({
      model: AVAILABLE_MODELS.GEMMA,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        {
          role: "user",
          content: "Explain how machine learning can be used in healthcare.",
        },
      ],
      temperature: 0.5,
      max_tokens: 200,
      stream: false,
    })) as ChatCompletionResponse;

    console.log("Response:", response.choices[0].message.content);
    console.log("Usage:", response.usage);
  } catch (error) {
    console.error("Error:", error);
  }

  // Example 4: Conversation with context
  console.log("\n--- Conversation with Context ---");
  try {
    const conversation = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is quantum computing?" },
      {
        role: "assistant",
        content:
          "Quantum computing is a type of computing that uses quantum mechanical phenomena like superposition and entanglement to perform calculations.",
      },
      { role: "user", content: "Can you give a practical example?" },
    ] as ChatMessage[];

    const response = (await client.chat.completions.create({
      model: AVAILABLE_MODELS.MIXTRAL,
      messages: conversation,
      temperature: 0.7,
      max_tokens: 150,
      stream: false,
    })) as ChatCompletionResponse;

    console.log("Response:", response.choices[0].message.content);
    console.log("Usage:", response.usage);
  } catch (error) {
    console.error("Error:", error);
  }
}

main().catch(console.error);
