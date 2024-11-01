# Anthropic Claude CLI Chat

A command-line interface for interacting with Anthropic's Claude AI models. This tool allows both interactive chat sessions and file-based interactions with Claude.

> Note: All code in this project was generated entirely by Claude (Anthropic). No human modifications were made to the generated code.

## Features

- Interactive chat mode with conversation history
- Support for multiline input
- File input/output capabilities
- Code generation with automatic markdown stripping
- Configurable model selection
- API key management
- Context preservation options

## Installation

1. Clone this repository
2. Install requirements:
pip install anthropic click

3. Set your Anthropic API key as an environment variable (optional):
export ANTHROPIC_API_KEY='your-api-key'

## Usage

### Basic Interactive Mode
python claude_cli.py

### Command Line Options
python claude_cli.py [OPTIONS]

Options:
  --api-key TEXT        Anthropic API Key (optional)
  --model TEXT          Anthropic model to use (default: claude-3-5-sonnet-latest)
  --no-context         Disable preserving conversation context
  --file PATH          Path to input file
  --prompt TEXT        Prompt to send to Claude
  --output PATH        Path to output file for saving responses
  --code-file PATH     Path to save generated code/content (strips markdown)
  --help              Show this message and exit

### Interactive Mode Commands
- Type 'exit' or press Ctrl+C to quit
- Type '/m' for multiline input mode
- Use '/save <filename>' to save generated code to a file

### Examples

1. Simple prompt with output to file:
python claude_cli.py --prompt "Explain quantum computing" --output quantum.txt

2. Generate code from a specification file:
python claude_cli.py --file spec.txt --code-file output.py

3. Interactive mode with context disabled:
python claude_cli.py --no-context

4. Use a specific model:
python claude_cli.py --model claude-3-opus-20240229

## Requirements

- Python 3.6+
- anthropic
- click

## Error Handling

The tool includes comprehensive error handling for:
- API communication issues
- File operations
- Invalid inputs
- Unexpected runtime errors