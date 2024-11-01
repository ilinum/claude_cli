from typing import List, Optional, Union
import os
import sys
import click
from anthropic import Anthropic
from pathlib import Path
import re

class ChatSession:
    def __init__(self, api_key: str, model: str, preserve_context: bool) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.preserve_context = preserve_context
        self.conversation_history: List[str] = []

    def send_message(self, message: str, context: Optional[str] = None) -> str:
        try:
            full_prompt = message
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion/Instruction:\n{message}"

            # Add script return instruction if script output is requested
            if '--script-output' in sys.argv:
                full_prompt += "\n\nPlease return the complete script in its entirety within triple backticks (```). Include all necessary imports, functions, and code blocks."

            if self.preserve_context and self.conversation_history:
                full_prompt = "\n".join(self.conversation_history + [full_prompt])

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": full_prompt}])
            ai_response = response.content[0].text
            if self.preserve_context:
                self.conversation_history.extend([full_prompt, ai_response])
            return ai_response
        except Exception as e:
            return f"An error occurred: {str(e)}"

def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text containing triple backticks."""
    # Pattern matches content between triple backticks, including language identifier
    pattern = r"```(?:\w*\n)?(.*?)```"
    # re.DOTALL allows matching across multiple lines
    matches = re.findall(pattern, text, re.DOTALL)
    # Strip leading/trailing whitespace from each block
    return [block.strip() for block in matches]

def validate_code_block(code: str) -> bool:
    """Basic validation of code block."""
    if not code:
        return False

    # Check for some basic Python syntax indicators
    basic_indicators = [
        'def ',
        'import ',
        'class ',
        '= ',
        'print(',
        'return ',
        'if ',
        'for ',
        'while '
    ]

    return any(indicator in code for indicator in basic_indicators)

def process_file(file_path: Path) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def save_to_file(content: str, file_path: Path) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        click.echo(f"Error saving to file: {str(e)}")

def handle_script_output(response: str, script_output: Path) -> None:
    """Handle extracting and saving code blocks from the response."""
    try:
        code_blocks = extract_code_blocks(response)

        if not code_blocks:
            click.echo("No valid code blocks found in the response.")
            return

        if len(code_blocks) == 1:
            # Single code block case
            code_content = code_blocks[0]
            if validate_code_block(code_content):
                save_to_file(code_content, script_output)
                click.echo(f"Script saved to {script_output}")
            else:
                click.echo("Extracted code block appears invalid.")
        else:
            # Multiple code blocks case
            valid_blocks = [block for block in code_blocks if validate_code_block(block)]

            if not valid_blocks:
                click.echo("No valid code blocks found in the response.")
                return

            if len(valid_blocks) == 1:
                save_to_file(valid_blocks[0], script_output)
                click.echo(f"Script saved to {script_output}")
            else:
                # Handle multiple valid code blocks
                base_path = script_output.parent
                stem = script_output.stem
                suffix = script_output.suffix

                for i, block in enumerate(valid_blocks, 1):
                    output_path = base_path / f"{stem}_{i}{suffix}"
                    save_to_file(block, output_path)
                    click.echo(f"Script part {i} saved to {output_path}")
    except Exception as e:
        click.echo(f"Error processing script output: {str(e)}")

@click.command()
@click.option('--api-key', help='Anthropic API Key (optional)')
@click.option('--model', default='claude-3-5-sonnet-latest', help='Anthropic model to use (default: claude-3-5-sonnet-latest)')
@click.option('--no-context', is_flag=True, help='Disable preserving conversation context')
@click.option('--file', type=click.Path(exists=True, path_type=Path), help='Path to input file')
@click.option('--prompt', help='Prompt to send to Claude')
@click.option('--output', type=click.Path(path_type=Path), help='Path to output file for saving responses')
@click.option('--script-output', type=click.Path(path_type=Path), help='Path to save generated script')
def main(api_key: Optional[str], model: str, no_context: bool, file: Optional[Path],
         prompt: Optional[str], output: Optional[Path], script_output: Optional[Path]) -> None:
    anthropic_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not anthropic_key:
        anthropic_key = click.prompt('Please enter your Anthropic API Key', hide_input=True)

    preserve_context = not no_context
    chat_session = ChatSession(api_key=anthropic_key, model=model, preserve_context=preserve_context)

    file_content = None
    if file:
        file_content = process_file(file)

    if prompt or file:
        if prompt and not file:
            # Only prompt provided
            response = chat_session.send_message(prompt)
        elif file and not prompt:
            # Only file provided
            response = chat_session.send_message(file_content)
        else:
            # Both file and prompt provided
            response = chat_session.send_message(prompt, context=file_content)

        if output:
            try:
                save_to_file(response, output)
                click.echo(f"Response saved to {output}")
            except Exception as e:
                click.echo(f"Error saving response: {str(e)}")
        else:
            click.echo(f"Claude: {response}")

        if script_output:
            handle_script_output(response, script_output)
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

                if script_output:
                    handle_script_output(response, script_output)
            except KeyboardInterrupt:
                break
    except Exception as e:
        click.echo(f"An unexpected error occurred: {str(e)}")
    click.echo("Goodbye!")

if __name__ == '__main__':
    main()