"""Record real local HTTP responses as an animated, Unicode-safe terminal transcript.

Install with: pip install -e '.[demo]'
Run with: python scripts/record_demo.py --preview-dir /tmp/vietnamese-demo-preview
The GIF replays responses; animation timing is for readability, not API latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata

import httpx
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 900
FONT_SIZE, LINE_HEIGHT = 24, 33
BACKGROUND = '#0d1117'
COLORS = {'text': '#e6edf3', 'command': '#79c0ff', 'comment': '#a6b3c2', 'result': '#7ee787'}


def clean_text(text: str) -> str:
    # Strict decoding alone cannot catch text that was already double-encoded.
    if any(marker in text for marker in ('\ufffd', 'Ã', 'Â', 'áº', 'á»')):
        raise ValueError('Possible encoding corruption; refusing to record')
    return unicodedata.normalize('NFC', text)


def choose_font(explicit: Path | None, text: str) -> tuple[Path, ImageFont.FreeTypeFont]:
    candidates = [explicit] if explicit else [
        Path('/System/Library/Fonts/Supplemental/Courier New.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),
        Path('C:/Windows/Fonts/consola.ttf'),
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        with TTFont(path, fontNumber=0) as face:
            cmap = face.getBestCmap()
            missing = {c for c in text if not c.isspace() and ord(c) not in cmap}
        if not missing:
            return path, ImageFont.truetype(str(path), FONT_SIZE)
    raise RuntimeError('No font covers all Vietnamese characters. Pass --font /path/to/font.ttf')


def wrap(text: str, font: ImageFont.FreeTypeFont) -> list[str]:
    lines: list[str] = []
    for paragraph in clean_text(text).split('\n'):
        line = ''
        for word in paragraph.split(' '):
            candidate = f'{line} {word}' if line else word
            if font.getlength(candidate) <= WIDTH - 88:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
                if font.getlength(line) > WIDTH - 88:
                    raise ValueError('Unbreakable text would overflow the recording')
        lines.append(line)
    return lines


def capture(cases: list[dict]) -> dict:
    env = os.environ.copy()
    env.pop('OPENROUTER_API_KEY', None)
    env['VIETNAMESE_RETRIEVER'] = 'lexical'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    # Pass our bound socket to uvicorn so no other process can steal its port.
    with socket.socket() as listener, tempfile.TemporaryFile() as log:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'vietnamese_rag.api:app', '--app-dir', str(ROOT / 'src'),
             '--fd', str(listener.fileno()), '--log-level', 'warning'],
            cwd=ROOT, env=env, pass_fds=(listener.fileno(),), stdout=log, stderr=log,
        )
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=30) as client:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        health = client.get('/health')
                        health.raise_for_status()
                        break
                    except httpx.HTTPError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise RuntimeError(log.read().decode('utf-8', errors='replace'))
                        time.sleep(0.1)

                def request(path: str, payload: dict) -> dict:
                    response = client.post(path, json=payload)
                    response.raise_for_status()
                    raw = clean_text(response.content.decode('utf-8', errors='strict'))
                    body = json.loads(raw)
                    if 'query' in payload and body['query'] != payload['query']:
                        raise ValueError('Vietnamese query did not survive the HTTP round trip')
                    return {'endpoint': path, 'request': payload, 'response': body}

                requests = [request('/query', {'query': 'tính 2+3*4', 'top_k': 3})]
                tool = requests[0]['response']
                if tool['route'] != 'tool' or tool['tool_trace'][0]['result']['result'] != 14:
                    raise ValueError('Calculator demo failed')
                for case in cases[:2]:
                    record = request('/query', {'query': case['query'], 'top_k': 3})
                    result = record['response']
                    ids = {doc['id'] for doc in result['retrieved_documents']}
                    if result['status'] != 'OK' or not result['citations'] or not ids.intersection(case['expected_doc_ids']):
                        raise ValueError('RAG demo did not retrieve the expected Vietnamese document')
                    requests.append(record)
                evaluation = request('/evaluate', {'top_k': 3, 'include_details': True})
                if evaluation['response']['cases'] != len(cases):
                    raise ValueError('Evaluation did not run the complete dataset')
                return {'mode': 'offline lexical + extractive', 'requests': requests, 'evaluation': evaluation}
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def scenes(record: dict, font: ImageFont.FreeTypeFont) -> list[list[tuple[str, str]]]:
    result = []
    for item in record['requests'] + [record['evaluation']]:
        body = item['response']
        rows = [('comment', '# Local HTTP response | Vietnamese dataset | offline')]
        rows += [('command', f'$ POST {item["endpoint"]}'), ('text', json.dumps(item['request'], ensure_ascii=False))]
        rows += [('comment', ''), ('result', 'HTTP 200 OK')]
        if item['endpoint'] == '/query':
            rows += [('result', f'route: {body["route"]}   status: {body["status"]}   generator: {body["generator"]}')]
            rows += [('comment', ''), ('comment', 'answer:'), ('text', body['answer'])]
            rows += [('comment', ''), ('comment', 'citations:'), ('text', json.dumps(body['citations'], ensure_ascii=False))]
            if body['tool_trace']:
                trace = body['tool_trace'][0]
                rows += [('comment', ''), ('text', f'tool: {trace["tool"]}   result: {trace["result"]["result"]}')]
        else:
            rows += [('comment', '# All cases from data/eval.jsonl; lexical heuristic metrics')]
            rows += [('text', f'{key}: {body[key]}') for key in (
                'cases', 'top_k', 'retrieval_hit_rate', 'retrieval_mrr_at_k',
                'citation_coverage', 'faithfulness', 'answer_relevancy',
            )]
        lines = [(kind, line) for kind, text in rows for line in wrap(text, font)]
        if len(lines) > 22:
            raise ValueError(f'Scene has {len(lines)} lines; would clip. Adjust layout before recording.')
        result.append(lines)
    return result


def render(rows: list[tuple[str, str]], font: ImageFont.FreeTypeFont, scene: int) -> Image.Image:
    image = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 64), fill='#161b22')
    for x, color in ((28, '#ff5f57'), (52, '#febc2e'), (76, '#28c840')):
        draw.ellipse((x, 26, x + 12, 38), fill=color)
    draw.text((116, 18), 'Vietnamese RAG / API session', font=font, fill=COLORS['text'])
    for index, (kind, text) in enumerate(rows):
        draw.text((44, 94 + LINE_HEIGHT * index), text, font=font, fill=COLORS[kind])
    draw.line((44, 840, WIDTH - 44, 840), fill='#30363d')
    draw.text((44, 850), f'{scene}/4  |  Real responses; fields selected for display', font=font, fill=COLORS['comment'])
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    datasets = [ROOT / 'data/knowledge_base.jsonl', ROOT / 'data/eval.jsonl']
    corpus = clean_text('\n'.join(path.read_text(encoding='utf-8') for path in datasets))
    cases = [json.loads(line) for line in datasets[1].read_text(encoding='utf-8').splitlines() if line.strip()]
    path, font = choose_font(args.font, corpus + 'tính 2+3*4 Kết quả: 14.')
    print(f'Preflight OK: strict UTF-8, font covers every dataset character ({path.name})', flush=True)
    record = capture(cases)
    record['datasets'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in datasets}
    record['font'] = path.name
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + '\n'
    choose_font(path, clean_text(serialized))
    all_scenes = scenes(record, font)
    frames, durations, final_indices = [], [], []
    for scene, rows in enumerate(all_scenes, 1):
        for count in range(1, len(rows) + 1):
            frame = render(rows[:count], font, scene).quantize(colors=128)
            duration = 8500 if count == len(rows) else 100
            if frames and frame.convert('RGB').tobytes() == frames[-1].convert('RGB').tobytes():
                durations[-1] += duration
            else:
                frames.append(frame)
                durations.append(duration)
        final_indices.append(len(frames) - 1)
    # Validate the encoded artifact before replacing the checked-in demo.
    with tempfile.TemporaryDirectory(prefix='vietnamese-recording-') as directory:
        gif = Path(directory) / 'demo.gif'
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
        with Image.open(gif) as decoded:
            if decoded.n_frames < 4 or decoded.info.get('loop') != 0:
                raise ValueError('Expected a looping, multi-frame GIF')
            for index, frame_index in enumerate(final_indices, 1):
                decoded.seek(frame_index)
                actual = decoded.convert('RGB')
                if actual.tobytes() != frames[frame_index].convert('RGB').tobytes():
                    raise ValueError('GIF encoding changed the rendered pixels')
                if args.preview_dir:
                    args.preview_dir.mkdir(parents=True, exist_ok=True)
                    actual.save(args.preview_dir / f'scene-{index}.png')
            print(f'GIF verified: {decoded.n_frames} frames, {sum(durations) / 1000:.1f}s, 4 scenes')
        (ROOT / 'docs/demo.gif').write_bytes(gif.read_bytes())
    (ROOT / 'docs/demo-responses.json').write_text(serialized, encoding='utf-8')
    print(json.dumps({key: record['evaluation']['response'][key] for key in (
        'cases', 'retrieval_hit_rate', 'retrieval_mrr_at_k', 'citation_coverage', 'faithfulness', 'answer_relevancy',
    )}, ensure_ascii=False))


if __name__ == '__main__':
    main()
