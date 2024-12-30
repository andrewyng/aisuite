# Chat UI

This is a simple chat UI built using Streamlit. It uses the `aisuite` library to power the chat.

You will need to install streamlit to run this example.

```bash
pip install streamlit
```

You will also need to create a `config.yaml` file in the same directory as the `chat.py` file. An example config file has been provided. You need to set environment variables for the API keys and other configuration for the LLMs you want to use. Place a .env file in this directory since `chat.py` will look for it.

In config.yaml, you can specify the LLMs you want to use in the chat. The chat UI will then display all these LLMs and you can select the one you want to use.

To run the app, simply run the following command in your terminal:

```bash
streamlit run chat.py
```

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
