# Chat UI

This is a simple chat UI built using Gradio and Streamlit. It uses the `aisuite` library to power the chat.

## Gradio Version

For the Gradio version, you'll need to install gradio:

```bash
pip install gradio
```

To run the Gradio app:

```bash
python gradio-chat.py
```

The Gradio version offers:
- A modern, clean interface
- Side-by-side comparison of two LLMs
- Easy toggling between single and comparison modes

You will need to install streamlit to run this example.

```bash
pip install streamlit
```

To run the app, simply run the following command in your terminal:

```bash
streamlit run chat.py
```

You will also need to create a `config.yaml` file in the same directory as the `chat.py` file. An example config file has been provided. You need to set environment variables for the API keys and other configuration for the LLMs you want to use. Place a .env file in this directory since both `gradio-chat.py` and `chat.py`  will look for it.

In config.yaml, you can specify the LLMs you want to use in the chat. The chat UI will then display all these LLMs and you can select the one you want to use.

You can choose different LLMs by ticking the "Comparison Mode" checkbox. Then select the two LLMs you want to compare.
Here are some sample queries you can try:

```
User: "What is the weather in Tokyo?"
```

```
User: "Write a poem about the weather in Tokyo."
```

```
User: "Write a python program to print the fibonacci sequence."
Assistant: "-- Content from LLM 1 --"
User: "Write test cases for this program."
```
