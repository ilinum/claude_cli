import os
import sys
import click
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class ChatSession:
    def __init__(self, api_key, model, preserve_context):
        """
        Initialize a chat session with Anthropic API.
        
        :param api_key: Anthropic API key
        :param model: Model to use for chat
        :param preserve_context: Whether to maintain conversation context
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.preserve_context = preserve_context
        self.conversation_history = []
        
    def send_message(self, message):
        """
        Send a message to the Anthropic API and return the response.
        
        :param message: User's input message
        :return: AI's response
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
            
            # Extract and store the response
            ai_response = response.content[0].text
            
            # Update conversation history if preserving context
            if self.preserve_context:
                self.conversation_history.extend([message, ai_response])
            
            return ai_response
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

@click.command()
@click.option('--api-key', help='Anthropic API Key (optional)')
@click.option('--model', default='claude-3-5-sonnet-20240620', 
              help='Anthropic model to use (default: latest Sonnet)')
@click.option('--no-context', is_flag=True, 
              help='Disable preserving conversation context')
def main(api_key, model, no_context):
    """
    Anthropic CLI Chat Application
    
    Interactive CLI for conversing with Anthropic's AI models.
    """
    # Determine API key
    if api_key:
        anthropic_key = api_key
    else:
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    # Prompt for API key if not provided
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
    
    # Welcome message
    click.echo(f"Welcome to Anthropic CLI Chat (Model: {model})")
    click.echo("Type 'exit' or press Ctrl+C to quit.")
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input("You: ")
                
                # Exit condition
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                # Get and display response
                response = chat_session.send_message(user_input)
                click.echo(f"Claude: {response}")
                
            except KeyboardInterrupt:
                break
    
    except Exception as e:
        click.echo(f"An unexpected error occurred: {str(e)}")
    
    click.echo("Goodbye!")

if __name__ == '__main__':
    main()
