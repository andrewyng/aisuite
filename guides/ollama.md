# Ollama

Ollama allows users to locally host open-source models available in [their library](https://ollama.com/library). 
Once an Ollama instance is locally running in your setup (default `http://localhost:11434`), you can use the `aisuite` API for chat completions as shown below.
No API Key is needed for these locally hosted models.

## Create a Chat Completion

Sample code:
```python
import aisuite as ai

def main():
    client = ai.Client(
        provider_configs={
            "ollama": {
                "api_url": "http://10.168.0.177:11434",
                "timeout": 300,
            }
        }
    )
    messages = [
        {
            "role": "system", 
            "content": "Be verbose"
        },
        {
            "role": "user", 
            "content": "Tell me something about University of Michigan's CSE department."
        },
    ]

    ollama_llama3 = "ollama:llama3:latest"
    ollama_gemma = "ollama:gemma:latest"
    ollama_deepseek_32B = "ollama:deepseek-r1:32b"
    ollama_deepseek_70B = "ollama:deepseek-r1:70b"

    response = client.chat.completions.create(
        model=ollama_gemma, 
        messages=messages, 
        temperature=0.75,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
