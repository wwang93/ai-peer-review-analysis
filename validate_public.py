"""Check the release allowlist, links, aggregate counts and topic weights."""
import csv
import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    '.gitignore', '.nojekyll', 'README.md', 'report.md', 'index.html', 'site.css',
    'prepare_data.py', 'model.py', 'plot_results.py', 'build_site.py',
    'validate_public.py', 'requirements.txt', 'run_config.json', 'topic_labels.json',
    'topics.csv', 'topic_by_round.csv', 'paired_changes.csv', 'sensitivity.csv',
    'model_candidates.csv', 'analysis_denominators.csv', 'provisional_codebook.csv',
    *{f'{name}.{ext}' for name in ['01_topic_patterns', '02_response_coverage',
       '03_paired_changes', '04_model_diagnostics', '05_prompt_change'] for ext in ['png', 'svg']},
}


def rows(name):
    with (ROOT / name).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images = set(), [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, 'Duplicate HTML ID'
            self.ids.add(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        if tag == 'img':
            assert attrs.get('alt'), 'Missing image description'
            self.images.append(attrs['src'])


def validate():
    present = {p.name for p in ROOT.iterdir() if p.is_file()}
    assert present == FILES, f'Public file allowlist differs: {present ^ FILES}'
    for name in FILES:
        path = ROOT / name
        if path.suffix in {'.md', '.html', '.csv', '.json', '.svg'}:
            text = path.read_text(encoding='utf-8')
            assert not re.search(r'\bS\d{2}(?:_R\d_Q\d|[-_]R\d)?\b|\bP\d{2}\b', text), f'Participant reference in {name}'
            assert '/Users/' not in text, f'Local path in {name}'
            assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}', text), f'Credential-like string in {name}'
        if path.suffix == '.csv':
            header = next(csv.reader(path.open()))
            assert not set(header) & {'student_id', 'response_id', 'original_text', 'model_text', 'text', 'example_ids', 'example_id', 'example_quote', 'source'}, f'Row-level field in {name}'
    for name in ['report.md', 'README.md']:
        text = (ROOT / name).read_text()
        assert '\u2014' not in text and '\u2013' not in text, 'Prose dash style check'
    page = Page()
    document = (ROOT / 'index.html').read_text()
    page.feed(document)
    assert len(page.images) == 5
    assert '{{' not in document
    for target in page.links + page.images:
        if target.startswith(('https://', 'http://')) or target == '#':
            continue
        if target.startswith('#'):
            assert target[1:] in page.ids, f'Broken anchor {target}'
        else:
            assert target in FILES, f'Nonpublic or missing local target {target}'
    totals = defaultdict(float)
    for row in rows('topic_by_round.csv'):
        value = float(row['mean_weight'])
        assert 0 <= value <= 1
        totals[(row['corpus'], row['round'])] += value
    assert all(math.isclose(value, 1, abs_tol=1e-10) for value in totals.values())
    denominators = rows('analysis_denominators.csv')
    main = [r for r in denominators if r['语料'] != 'R4_affect']
    assert sum(int(r['应有']) for r in main) == 429
    assert sum(int(r['有效学生文本']) for r in main) == 393
    assert sum(int(r['模型文本']) for r in main) == 386
    assert sum(int(r['有效学生文本']) for r in denominators) == 427
    assert len(rows('provisional_codebook.csv')) == 36
    assert set(json.loads((ROOT / 'topic_labels.json').read_text())) == {'RQ1', 'RQ2', 'RQ3', 'RQ3_extended'}
    print(f'Validated {len(FILES)} public files, five figures, 36 candidate codes, links and aggregate denominators.')


if __name__ == '__main__':
    validate()
