# Reddit Research

Tools for gathering and analyzing Reddit data using LLMs - much simplified version of the [Reddit Answers](https://www.reddit.com/answers).

Usage:
```uv run research.py "my initial prompt"```

Be sure to set the environment variable `GEMINI_API_KEY` to your Gemini API key.
Create config.py with your [Reddit API credentials](https://redditwarp.readthedocs.io/en/latest/getting-started/authorization.html):
```python
CONFIG = {
"app_id": ...,
"secret": ...,
"refresh_token": ...
}
```

To save reports as PDFs, also install (macos only):
```
brew install pandoc
brew install --cask basictex
```

Typical usage with Gemini Flash takes ~5 minutes to generate a report and ~$0.5 in costs, with 200-300 pages analyzed. Numbers are very variable depending on the prompt.

It also works ok-ish with local models (e.g. ollama/qwen3:30b), but takes hours on a laptop. Don't forget to change Ollama's default max context size (e.g. with env var `OLLAMA_CONTEXT_LENGTH=32768`) for acceptable quality.

### Disclaimer

This project is intended solely for educational and research purposes.
