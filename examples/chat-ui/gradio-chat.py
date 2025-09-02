import os
import gradio as gr
import yaml
from dotenv import load_dotenv, find_dotenv
from aisuite.client import Client

# Load configuration and initialize aisuite client
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
configured_llms = config["llms"]
load_dotenv(find_dotenv())
client = Client()

def query_llm(model_config, chat_history):
    print(f"Querying {model_config['name']} with {chat_history}")
    try:
        model = model_config["provider"] + ":" + model_config["model"]
        response = client.chat.completions.create(model=model, messages=chat_history)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying {model_config['name']}: {e}"

def format_message(user, message):
    return f"{'User' if user else 'Assistant'}: {message}"

def chat(message, history, model_name):
    if not message:
        return "", history

    # Add user message to history
    history.append((message, None))
    model_config = next(llm for llm in configured_llms if llm["name"] == model_name)
    
    # Format chat history for API
    chat_messages = []
    for user_msg, assistant_msg in history:
        chat_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            chat_messages.append({"role": "assistant", "content": assistant_msg})

    # Get response from LLM
    response = query_llm(model_config, chat_messages)
    
    # Update history with assistant's response
    history[-1] = (message, response)
    
    return "", history

def reset_chat():
    return [], []

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chat & Compare LLM responses")
    
    with gr.Row():
        comparison_mode = gr.Checkbox(label="Comparison Mode", value=False)
    
    with gr.Row():
        model1_dropdown = gr.Dropdown(
            choices=[llm["name"] for llm in configured_llms],
            value=configured_llms[0]["name"],
            label="Choose LLM Model 1"
        )
        model2_dropdown = gr.Dropdown(
            choices=[llm["name"] for llm in configured_llms],
            value=configured_llms[1]["name"] if len(configured_llms) > 1 else configured_llms[0]["name"],
            label="Choose LLM Model 2",
            visible=False
        )

    with gr.Row() as chat_row:
        with gr.Column(scale=1) as col1:
            chatbot1 = gr.Chatbot(label="Chat 1")
            with gr.Row():
                msg1 = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                )
                clear1 = gr.Button("Clear")
        
        with gr.Column(scale=1, visible=False) as col2:
            chatbot2 = gr.Chatbot(label="Chat 2")
            with gr.Row():
                msg2 = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                )
                clear2 = gr.Button("Clear")

    # Event handlers
    msg1.submit(chat, [msg1, chatbot1, model1_dropdown], [msg1, chatbot1])
    clear1.click(reset_chat, outputs=[chatbot1, msg1])
    
    msg2.submit(chat, [msg2, chatbot2, model2_dropdown], [msg2, chatbot2])
    clear2.click(reset_chat, outputs=[chatbot2, msg2])

    # Toggle comparison mode
    def toggle_comparison(checkbox_value):
        return {
            model2_dropdown: gr.update(visible=checkbox_value),
            col2: gr.update(visible=checkbox_value),
            col1: gr.update(scale=2 if not checkbox_value else 1)
        }

    # Update the comparison_mode.change event
    comparison_mode.change(
        toggle_comparison,
        inputs=[comparison_mode],
        outputs=[model2_dropdown, col2, col1]
    )

if __name__ == "__main__":
    demo.launch()
