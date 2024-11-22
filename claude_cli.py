from typing import List, Optional
import os
import click
from anthropic import Anthropic
import re
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from colorama import init, Fore, Style as ColoramaStyle
from datetime import datetime
import json

# Initialize colorama
init()

HISTORY_FILE = os.path.expanduser('~/.claudehistory')

prompt_style = Style.from_dict({
    'prompt': '#0000FF',  # Blue
    'input': '#FFFFFF',   # White
})

class ChatSession:
    def __init__(self, api_key: str, model: str, preserve_context: bool) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.preserve_context = preserve_context
        self.conversation_history: List[str] = []
        self.session = PromptSession(style=prompt_style)

    def send_message(self, message: str, context: Optional[str] = None, code_output: bool = False, stream: bool = False) -> str:
        try:
            full_prompt = message
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion/Instruction:\n{message}"

            if code_output:
                full_prompt = f"{full_prompt}\n\nPlease provide the complete code file/content. Do not include any explanations or markdown formatting - return only the actual file content that should be saved."

            if self.preserve_context and self.conversation_history:
                full_prompt = "\n".join(self.conversation_history + [full_prompt])

            if stream:
                response_text = ""
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": full_prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        response_text += text
                        print(text, end='', flush=True)
                    print()  # New line after stream ends
                ai_response = response_text
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": full_prompt}])
                ai_response = response.content[0].text

            if code_output:
                # Remove markdown code blocks if present
                ai_response = re.sub(r'^```[\w]*\n|```$', '', ai_response, flags=re.MULTILINE).strip()

            if self.preserve_context:
                self.conversation_history.extend([full_prompt, ai_response])
                
            # Save to history
            save_to_history(message, ai_response, self.model)
            
            return ai_response
        except Exception as e:
            return f"An error occurred: {str(e)}"

def process_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def save_to_file(content: str, file_path: str) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        click.echo(f"Error saving to file: {str(e)}")

def save_to_history(prompt: str, response: str, model: str) -> None:
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt': prompt,
            'response': response
        })
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        click.echo(f"Error saving to history: {str(e)}")

def get_multiline_input(session: PromptSession) -> str:
    """Get multiline input from the user until they press Ctrl+D/Z or enter two blank lines."""
    click.echo(f"{Fore.YELLOW}Enter your message (press Ctrl+D or Ctrl+Z to finish, or enter two blank lines):{ColoramaStyle.RESET_ALL}")
    lines = []
    blank_line_count = 0
    
    while True:
        try:
            line = session.prompt('... ')
            if not line:
                blank_line_count += 1
                if blank_line_count >= 2:
                    break
            else:
                blank_line_count = 0
            lines.append(line)
        except EOFError:
            break
    
    return '\n'.join(lines).strip()

@click.command()
@click.option('--api-key', help='Anthropic API Key (optional)')
@click.option('--model', default='claude-3-5-sonnet-latest', help='Anthropic model to use (default: claude-3-5-sonnet-latest)')
@click.option('--no-context', is_flag=True, help='Disable preserving conversation context')
@click.option('--file', type=click.Path(exists=True), help='Path to input file')
@click.option('--prompt', help='Prompt to send to Claude')
@click.option('--output', type=click.Path(), help='Path to output file for saving responses')
@click.option('--code-file', type=click.Path(), help='Path to save generated code/content (strips markdown)')
def main(api_key: Optional[str], model: str, no_context: bool, file: Optional[str],
         prompt: Optional[str], output: Optional[str], code_file: Optional[str]) -> None:
    
    anthropic_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not anthropic_key:
        anthropic_key = click.prompt('Please enter your Anthropic API Key', hide_input=True)

    preserve_context = not no_context
    chat_session = ChatSession(api_key=anthropic_key, model=model, preserve_context=preserve_context)

    file_content = None
    if file:
        file_content = process_file(file)

    if prompt or file:
        code_output = bool(code_file)
        if prompt and not file:
            response = chat_session.send_message(prompt, code_output=code_output)
        elif file and not prompt:
            response = chat_session.send_message(file_content, code_output=code_output)
        else:
            response = chat_session.send_message(prompt, context=file_content, code_output=code_output)

        if code_file:
            save_to_file(response, code_file)
            click.echo(f"Code saved to {code_file}")
        if output:
            save_to_file(response, output)
            click.echo(f"Response saved to {output}")
        if not output and not code_file:
            click.echo(f"Claude: {response}")
        return

    # Interactive mode
    click.echo(f"{Fore.CYAN}Welcome to Anthropic CLI Chat (Model: {model}){ColoramaStyle.RESET_ALL}")
    click.echo(f"{Fore.GREEN}Type 'exit' or press Ctrl+C to quit.{ColoramaStyle.RESET_ALL}")
    click.echo(f"{Fore.GREEN}Type '/m' for multiline input mode{ColoramaStyle.RESET_ALL}")
    click.echo(f"{Fore.GREEN}To save code, use: /save <filename>{ColoramaStyle.RESET_ALL}")

    try:
        while True:
            try:
                user_input = chat_session.session.prompt(f"{Fore.BLUE}You: {ColoramaStyle.RESET_ALL}")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break

                if user_input == '/m':
                    user_input = get_multiline_input(chat_session.session)
                    if not user_input:  # Skip empty input
                        continue

                if user_input.startswith('/save '):
                    filename = user_input.split(' ', 1)[1]
                    prompt = get_multiline_input(chat_session.session)
                    if prompt:
                        response = chat_session.send_message(prompt, code_output=True)
                        save_to_file(response, filename)
                        click.echo(f"{Fore.GREEN}Code saved to {filename}{ColoramaStyle.RESET_ALL}")
                else:
                    print(f"{Fore.MAGENTA}Claude: {ColoramaStyle.RESET_ALL}", end='', flush=True)
                    response = chat_session.send_message(user_input, stream=True)
            except KeyboardInterrupt:
                break
    except Exception as e:
        click.echo(f"{Fore.RED}An unexpected error occurred: {str(e)}{ColoramaStyle.RESET_ALL}")
    click.echo(f"{Fore.CYAN}Goodbye!{ColoramaStyle.RESET_ALL}")

if __name__ == '__main__':
    main()