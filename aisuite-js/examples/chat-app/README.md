# AISuite Chat App

A modern React TypeScript chat application built with AISuite, allowing you to chat with multiple AI models and compare their responses in real-time.

## Features

- **Multi-Provider Support**: Chat with OpenAI, Anthropic, Groq, and Mistral models
- **Comparison Mode**: Compare responses from two different AI models side-by-side
- **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- **Real-time Chat**: Instant messaging with AI models
- **API Key Management**: Secure storage and management of API keys
- **Error Handling**: Comprehensive error handling and user feedback
- **TypeScript**: Full type safety throughout the application

## Prerequisites

- Node.js 18+ 
- npm or yarn
- API keys for the AI providers you want to use:
  - OpenAI API key
  - Anthropic API key
  - Groq API key
  - Mistral API key

## Installation

1. Clone the repository and navigate to the chat app directory:
```bash
cd aisuite-js/chat-app
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:3000`

## Configuration

### API Keys

1. Click the "Configure API Keys" button in the header
2. Enter your API keys for the providers you want to use
3. Click "Save" to store the configuration

The app will automatically save your API keys to localStorage for future use.

### Supported Models

The app comes pre-configured with the following models:

**OpenAI:**
- GPT-4o
- GPT-4o Mini

**Anthropic:**
- Claude 3.5 Sonnet
- Claude 3 Haiku

**Groq:**
- Llama 3.1 8B
- Mixtral 8x7B

**Mistral:**
- Mistral 7B
- Mistral Large

## Usage

### Basic Chat

1. Configure your API keys
2. Select a model from the dropdown
3. Type your message and press Enter or click Send
4. View the AI response

### Comparison Mode

1. Enable "Comparison Mode" checkbox
2. Select two different models
3. Send a message to see responses from both models side-by-side
4. Compare the different responses and capabilities

### Chat Management

- **Reset Chat**: Click the reset button to clear all chat history
- **Model Switching**: Change models at any time during the conversation
- **Error Handling**: The app displays clear error messages for API issues

## Sample Queries

Try these sample queries to test the different models:

```
"What is the weather in Tokyo?"
```

```
"Write a poem about the weather in Tokyo."
```

```
"Write a python program to print the fibonacci sequence."
```

```
"Write test cases for this program."
```

## Development

### Project Structure

```
src/
├── components/          # React components
│   ├── ApiKeyModal.tsx
│   ├── ChatContainer.tsx
│   ├── ChatInput.tsx
│   ├── ChatMessage.tsx
│   └── ModelSelector.tsx
├── config/             # Configuration files
│   └── llm-config.ts
├── services/           # Business logic
│   └── aisuite-service.ts
├── types/              # TypeScript type definitions
│   └── chat.ts
├── App.tsx            # Main application component
├── main.tsx           # Application entry point
└── index.css          # Global styles
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Adding New Models

To add new models, edit `src/config/llm-config.ts`:

```typescript
export const configuredLLMs: LLMConfig[] = [
  // ... existing models
  {
    name: "Your New Model",
    provider: "provider-name",
    model: "model-name"
  }
];
```

### Styling

The app uses Tailwind CSS for styling. The design system includes:

- Light and dark mode support
- Responsive design
- Custom scrollbars
- Loading animations
- Error states

## Technologies Used

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **AISuite** - AI provider abstraction

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see the main repository for details.

## Support

For issues and questions:
- Check the [AISuite documentation](https://github.com/andrewyng/aisuite)
- Open an issue in the repository
- Check the console for error messages

## Security Notes

- API keys are stored in localStorage (client-side only)
- No API keys are sent to any server except the AI providers
- Consider using environment variables for production deployments 