import "dotenv/config";
import { Client, ChatCompletionResponse, ChatMessage } from "../src";

// Mock function for weather
function getWeather(location: string, unit: 'celsius' | 'fahrenheit' = 'celsius') {
  // Mock implementation
  return {
    location,
    temperature: unit === 'celsius' ? 22 : 72,
    condition: 'sunny',
    unit
  };
}

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

  // Example 5: Tool calling with Groq
  console.log("\n--- Tool Calling Example with Groq ---");
  try {
    // Define tools in OpenAI format
    const tools = [
      {
        type: 'function' as const,
        function: {
          name: 'get_weather',
          description: 'Get the current weather for a location',
          parameters: {
            type: 'object' as const,
            properties: {
              location: {
                type: 'string',
                description: 'The city and state, e.g. San Francisco, CA'
              },
              unit: {
                type: 'string',
                enum: ['celsius', 'fahrenheit'],
                description: 'The temperature unit'
              }
            },
            required: ['location']
          }
        }
      }
    ];

    // Step 1: Initial request with tools
    const response = (await client.chat.completions.create({
      model: AVAILABLE_MODELS.MIXTRAL,
      messages: [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in London?" }
      ],
      tools,
      tool_choice: 'auto'
    })) as ChatCompletionResponse;

    const message = response.choices[0]?.message;
    console.log('Step 1 - Initial response:', JSON.stringify(message, null, 2));

    if (message?.tool_calls) {
      // Step 2: Execute tool calls and send results back
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in London?" },
        message // The assistant's message with tool calls
      ];

      console.log('\nTool calls detected:');
      for (const toolCall of message.tool_calls) {
        console.log(`- Function: ${toolCall.function.name}`);
        console.log(`  Arguments: ${toolCall.function.arguments}`);
        
        // Execute the function
        const args = JSON.parse(toolCall.function.arguments);
        const result = getWeather(args.location, args.unit);
        console.log(`  Result:`, result);

        // Add tool result to messages
        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(result)
        });
      }

      // Step 3: Get final response with tool results
      console.log('\nStep 2 - Sending tool results back...');
      const finalResponse = (await client.chat.completions.create({
        model: AVAILABLE_MODELS.MIXTRAL,
        messages,
        temperature: 0.7,
        max_tokens: 200
      })) as ChatCompletionResponse;

      console.log('\nStep 3 - Final response:', finalResponse.choices[0].message.content);
    }
  } catch (error) {
    console.error("Tool calling error:", error);
  }
}

main().catch(console.error);
