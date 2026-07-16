import "dotenv/config";
import { Client, ChatCompletionResponse, ChatMessage } from "../src";

// Sample data store
const data = {
  transactionId: ["T1001", "T1002", "T1003", "T1004", "T1005"],
  customerId: ["C001", "C002", "C003", "C002", "C001"],
  paymentAmount: [125.5, 89.99, 120.0, 54.3, 210.2],
  paymentDate: [
    "2021-10-05",
    "2021-10-06",
    "2021-10-07",
    "2021-10-05",
    "2021-10-08",
  ],
  paymentStatus: ["Paid", "Unpaid", "Paid", "Paid", "Pending"],
};

/**
 * Retrieves the payment status for a given transaction
 */
function retrievePaymentStatus({ data, transactionId }) {
  const transactionIndex = data.transactionId.indexOf(transactionId);
  if (transactionIndex !== -1) {
    return JSON.stringify({ status: data.paymentStatus[transactionIndex] });
  }
  return JSON.stringify({ status: "error - transaction id not found" });
}

/**
 * Retrieves the payment date for a given transaction
 */
function retrievePaymentDate({ data, transactionId }) {
  const transactionIndex = data.transactionId.indexOf(transactionId);
  if (transactionIndex !== -1) {
    return JSON.stringify({ date: data.paymentDate[transactionIndex] });
  }
  return JSON.stringify({ date: "error - transaction id not found" });
}

// Map function names to their implementations
const namesToFunctions = {
  retrievePaymentStatus: (transactionId) =>
    retrievePaymentStatus({ data, ...transactionId }),
  retrievePaymentDate: (transactionId) =>
    retrievePaymentDate({ data, ...transactionId }),
};

// Define available tools (functions) for the model
const TOOLS = [
  {
    type: "function",
    function: {
      name: "retrievePaymentStatus",
      description: "Get payment status of a transaction id",
      parameters: {
        type: "object",
        required: ["transactionId"],
        properties: {
          transactionId: { type: "string", description: "The transaction id." },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "retrievePaymentDate",
      description: "Get payment date of a transaction id",
      parameters: {
        type: "object",
        required: ["transactionId"],
        properties: {
          transactionId: { type: "string", description: "The transaction id." },
        },
      },
    },
  },
];

async function main() {
  const client = new Client({
    mistral: { apiKey: process.env.MISTRAL_API_KEY! },
  });

  console.log("\n🔮 Mistral Chat Examples\n");

  // Example 1: Basic chat completion
  console.log("--- Basic Chat Completion ---");
  try {
    const response = (await client.chat.completions.create({
      model: "mistral:mistral-medium",
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

  // Example 2: Streaming
  console.log("\n--- Streaming Example ---");
  try {
    const stream = await client.chat.completions.create({
      model: "mistral:mistral-medium",
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
  } catch (error) {
    console.error("Streaming error:", error);
  }

  // Example 3: Tool calling
  console.log("\n\n--- Tool Calling Example ---");
  try {
    const tools = [
      {
        type: "function",
        function: {
          name: "retrievePaymentStatus",
          description: "Get payment status of a transaction id",
          parameters: {
            type: "object",
            required: ["transactionId"],
            properties: {
              transactionId: {
                type: "string",
                description: "The transaction id.",
              },
            },
          },
        },
      },
      {
        type: "function",
        function: {
          name: "retrievePaymentDate",
          description: "Get payment date of a transaction id",
          parameters: {
            type: "object",
            required: ["transactionId"],
            properties: {
              transactionId: {
                type: "string",
                description: "The transaction id.",
              },
            },
          },
        },
      },
    ];
    const model = "mistral:mistral-large-latest";

    let messages: ChatMessage[] = [
      { role: "user", content: "What's the status of my transaction?" },
    ];

    // First interaction - Model asks for transaction ID
    let response = (await client.chat.completions.create({
      model,
      messages: [
        { role: "user", content: "What's the status of my transaction?" },
      ],
      tools: TOOLS as any, // Type assertion for Mistral's string-based tools
    })) as ChatCompletionResponse;

    messages.push({
      role: "assistant",
      content: response.choices[0].message.content as string,
    });

    // User provides transaction ID
    messages.push({ role: "user", content: "My transaction ID is T1001." });

    // Second interaction - Model uses functions to get information
    response = (await client.chat.completions.create({
      model,
      messages,
      tools: TOOLS as any, // Type assertion for Mistral's string-based tools
    })) as ChatCompletionResponse;

    messages.push(response.choices[0].message);

    // Process tool calls
    const toolCalls = response.choices[0].message.tool_calls || [];
    for (const toolCall of toolCalls) {
      const functionName = toolCall.function.name;
      const functionParams = JSON.parse(toolCall.function.arguments);

      console.log(`Calling function: ${functionName}`);
      console.log(`Parameters: ${toolCall.function.arguments}`);

      const functionResult = namesToFunctions[functionName](functionParams);

      messages.push({
        role: "tool",
        name: functionName,
        content: functionResult,
        tool_call_id: toolCall.id,
      });
    }

    // Final response with the information
    response = (await client.chat.completions.create({
      model,
      messages,
      tools: TOOLS as any, // Type assertion for Mistral's string-based tools
    })) as ChatCompletionResponse;

    console.log("Final response:", response.choices[0].message.content);
  } catch (error) {
    console.error("Tool calling error:", error);
  }
}

main().catch(console.error);
