# AISuite

This is my fork of the original [aisuite](https://github.com/andrewyng/aisuite) repository. Everything remains the same except that a new **GitHub Provider** has been added to enable free GPT-4 access via a GitHub Copilot subscription. This addition is inspired by [freegpt](https://github.com/B00TK1D/freegpt).

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hubertusgbecker/aisuite.git
cd aisuite
pip install -e .
```

## Usage

Use the new GitHub Provider as follows:

```python
from aisuite.providers.github_provider import GitHubProvider
from aisuite.framework.message import Message

provider = GitHubProvider()
prompt = {"content": "Hello, what is the capital of Germany?", "role": "user"}
response = provider.chat_completions_create("gpt-4", [prompt])
print(response.choices[0].message.content)
```

## License

aisuite is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

