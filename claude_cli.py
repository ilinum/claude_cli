from typing import List, Optional, Union
import os
import sys
import click
from anthropic import Anthropic
from pathlib import Path

class ChatSession:
    def __init__(self, api_key: str, model: str, preserve_context: bool) -> None:
        """
        Initialize a chat session with Anthropic API.

        Args:
            api_key: Anthropic API key
            model: Model to use for chat
            preserve_context: Whether to maintain conversation context
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.preserve_context = preserve_context
        self.conversation_history: List[str] = []

    def send_message(self, message: str) -> str:
        """
        Send a message to the Anthropic API and return the response.

        Args:
            message: User's input message
        Returns:
            AI's response
        """
        try:
            # Prepare the context if preserving
            full_prompt = message
            if self.preserve_context and self.conversation_history:
                full_prompt = "\n".join(self.conversation_history + [message])

            # Send message to Anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            ai_response = response.content[0].text

            if self.preserve_context:
                self.conversation_history.extend([message, ai_response])

            return ai_response

        except Exception as e:
            return f"An error occurred: {str(e)}"

def process_file(file_path: Path) -> str:
    """
    Read and process a file's content.

    Args:
        file_path: Path to the file
    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@click.command()
@click.option('--api-key', help='Anthropic API Key (optional)')
@click.option('--model', default='claude-3-sonnet-latest',
              help='Anthropic model to use (default: claude-3-sonnet-latest)')
@click.option('--no-context', is_flag=True,
              help='Disable preserving conversation context')
@click.option('--file', type=click.Path(exists=True, path_type=Path),
              help='Path to input file')
@click.option('--output', type=click.Path(path_type=Path),
              help='Path to output file for saving responses')
def main(api_key: Optional[str], model: str, no_context: bool,
         file: Optional[Path], output: Optional[Path]) -> None:
    """
    Anthropic CLI Chat Application

    Interactive CLI for conversing with Anthropic's AI models.
    """
    # Determine API key
    anthropic_key = api_key or os.environ.get('ANTHROPIC_API_KEY')

    if not anthropic_key:
        anthropic_key = click.prompt(
            'Please enter your Anthropic API Key',
            hide_input=True
        )

    # Initialize chat session
    preserve_context = not no_context
    chat_session = ChatSession(
        api_key=anthropic_key,
        model=model,
        preserve_context=preserve_context
    )

    # Process file input if provided
    if file:
        content = process_file(file)
        response = chat_session.send_message(content)

        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(response)
                click.echo(f"Response saved to {output}")
            except Exception as e:
                click.echo(f"Error saving response: {str(e)}")
        else:
            click.echo(f"Claude: {response}")
        return

    # Interactive mode
    click.echo(f"Welcome to Anthropic CLI Chat (Model: {model})")
    click.echo("Type 'exit' or press Ctrl+C to quit.")

    try:
        while True:
            try:
                user_input = input("You: ")

                if user_input.lower() in ['exit', 'quit', 'q']:
                    break

                response = chat_session.send_message(user_input)
                click.echo(f"Claude: {response}")

            except KeyboardInterrupt:
                break

    except Exception as e:
        click.echo(f"An unexpected error occurred: {str(e)}")

    click.echo("Goodbye!")

if __name__ == '__main__':
    main()