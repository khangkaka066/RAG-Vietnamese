# Reproduce the API demo

`demo.gif` is an animated transcript rendered from real HTTP responses, not a screen
capture or the static SVG overview. It shows selected response fields verbatim;
the full payloads are saved in [demo-responses.json](demo-responses.json).
Frames appear line by line, with pauses for reading; timing does not measure API latency.

```bash
pip install -e '.[demo]'
python scripts/record_demo.py --preview-dir /tmp/vietnamese-demo-preview
```

Run from the repository root on macOS or Linux. The recorder starts its own uvicorn
process on a free localhost port and shuts it down afterward. It disables OpenRouter
credentials and uses lexical retrieval and extractive generation, with no model download.

The session calls the calculator, asks the first two questions from `data/eval.jsonl`
against `data/knowledge_base.jsonl`, then evaluates all 20 cases via `/evaluate`.
Dataset SHA-256 hashes are stored alongside the responses.

Before replacing the GIF, the recorder checks:

- Strict UTF-8 decoding and common encoding-corruption markers.
- Font glyph coverage for every dataset and response character, normalized to NFC.
- Exact query round trips, calculator output, expected retrieved documents, and citations.
- Text width and line count to prevent clipping.
- Multiple animation frames, infinite looping, and pixel-exact decoded final scenes.

Default fonts are Courier New on macOS and DejaVu Sans Mono on Linux.
To use another font, pass `--font /path/to/font.ttf`; missing Vietnamese glyphs cause
the recorder to stop before publishing. Final-scene PNGs in `--preview-dir` allow
visual inspection of the decoded GIF, including accents and citations.
