# Reddit Research

Tools for gathering and analyzing Reddit data using LLMs.
Usage:
```uv run research.py "my initial prompt"```

Be sure to set the environment variable `GEMINI_API_KEY` to your Gemini API key.
Create config.py with your Reddit API credentials:
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
