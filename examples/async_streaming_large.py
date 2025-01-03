"""
Example demonstrating streaming of large responses with progress tracking.
"""
import os
import sys
import asyncio
import time
import argparse

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from aisuite import AsyncClient
from aisuite.providers.async_openai_provider import OpenAIError

# Initialize Rich console for beautiful output
console = Console()

async def stream_large_response(api_key: str):
    """Stream a large response with progress tracking."""
    try:
        # Create a complex prompt that will generate a large response
        messages = [{
            "role": "user",
            "content": """Write a detailed technical analysis of the following topics:
            1. The evolution of artificial intelligence
            2. The impact of quantum computing on cryptography
            3. The future of autonomous vehicles
            4. The role of blockchain in finance
            5. The potential of brain-computer interfaces
            
            For each topic, include:
            - Historical background
            - Current state of technology
            - Future predictions
            - Technical challenges
            - Ethical considerations
            - Real-world applications
            
            Make it very detailed and comprehensive."""
        }]

        # Initialize client
        async with AsyncClient({
            "openai": {"api_key": api_key}
        }) as client:
            # Setup progress tracking
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Create a task for tracking
                task = progress.add_task("[cyan]Streaming response...", total=None)
                
                # Variables for tracking response
                start_time = time.time()
                chunk_count = 0
                total_tokens = 0
                current_section = ""
                full_content = []
                
                try:
                    # Stream the response using create with stream=True
                    stream = await client.chat.completions.create(
                        model="openai:gpt-4",
                        messages=messages,
                        temperature=1.0,
                        max_tokens=4000,
                        stream=True
                    )
                    
                    async for chunk in stream:
                        # Extract content from chunk
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            
                            # Update progress
                            chunk_count += 1
                            total_tokens += len(content.split())
                            full_content.append(content)
                            
                            # Check for new section
                            if "." in content:
                                current_section = content.split(".")[-1].strip()
                            
                            # Update progress description
                            progress.update(
                                task,
                                description=f"[cyan]Streaming response... Section: {current_section[:30]}..."
                            )
                            
                            # Print content with formatting
                            if content.strip():
                                # Format headings
                                if content.strip().isupper():
                                    console.print(content, style="bold cyan")
                                # Format subheadings
                                elif ":" in content:
                                    parts = content.split(":")
                                    if len(parts) > 1:
                                        console.print(
                                            f"{parts[0]}:",
                                            style="bold yellow",
                                            end=""
                                        )
                                        console.print(":".join(parts[1:]))
                                # Regular content
                                else:
                                    # Use regular print for streaming content
                                    print(content, end="", flush=True)
                    
                    # Calculate and display statistics
                    elapsed_time = time.time() - start_time
                    console.print("\n\n[bold green]Stream completed![/bold green]")
                    console.print(f"[yellow]Statistics:[/yellow]")
                    console.print(f"- Time elapsed: {elapsed_time:.2f} seconds")
                    console.print(f"- Total chunks: {chunk_count}")
                    console.print(f"- Approximate tokens: {total_tokens}")
                    console.print(f"- Average tokens per second: {total_tokens/elapsed_time:.2f}")
                    
                except Exception as e:
                    console.print(f"\n[bold red]Error during streaming:[/bold red] {str(e)}")
                    raise
                
    except OpenAIError as e:
        console.print(f"\n[bold red]OpenAI Error:[/bold red] {str(e)}")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stream a large response from OpenAI API")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    args = parser.parse_args()
    
    # Run the streaming example
    asyncio.run(stream_large_response(args.api_key))
