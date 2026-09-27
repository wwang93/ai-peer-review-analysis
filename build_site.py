"""Build the static report from Markdown and aggregate CSVs; no student text required."""
import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = 'https://github.com/wwang93/ai-peer-review-analysis'


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def md_table(headers, rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in [headers, ['---'] * len(headers), *rows])


def replace_table(text, key, value):
    block = f'<!-- {key}:start -->\n{value}\n<!-- {key}:end -->'
    pattern = rf'<!-- {key}:start -->.*?<!-- {key}:end -->'
    if re.search(pattern, text, re.S):
        return re.sub(pattern, lambda _: block, text, flags=re.S)
    return text.replace('{{' + key + '}}', block)


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text


def render_markdown(text):
    """Render the small Markdown subset used by this report."""
    lines = text.splitlines()
    output, nav, paragraph = [], [], []
    section = 0

    def flush():
        if paragraph:
            output.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('<!--'):
            flush()
        elif line.startswith('# '):
            flush()  # The title is rendered in the page header.
        elif line.startswith('## '):
            flush()
            section += 1
            title = line[3:]
            nav.append(f'<a href="#section-{section}"><span>{section:02}</span>{html.escape(title)}</a>')
            output.append(f'<h2 id="section-{section}">{html.escape(title)}</h2>')
        elif line.startswith('### '):
            flush()
            output.append('<h3>' + inline(line[4:]) + '</h3>')
        elif line.startswith('!['):
            flush()
            match = re.fullmatch(r'!\[([^]]+)\]\(([^)]+)\)', line)
            alt, path = match.groups()
            svg = path.replace('.png', '.svg')
            output.append(f'<figure><a href="{path}" aria-label="查看大图：{html.escape(alt)}"><img src="{path}" alt="{html.escape(alt)}" loading="lazy" width="1200" height="800"></a><figcaption>{html.escape(alt)} <a href="{svg}" download>下载 SVG</a></figcaption></figure>')
        elif line.startswith('|'):
            flush()
            table = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    table.append(cells)
                i += 1
            output.append('<div class="table-scroll" role="region" tabindex="0" aria-label="数据表格，可横向滚动"><table><thead><tr>' + ''.join('<th scope="col">' + inline(c) + '</th>' for c in table[0]) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in row) + '</tr>' for row in table[1:]) + '</tbody></table></div>')
            continue
        else:
            paragraph.append(line)
        i += 1
    flush()
    return '\n'.join(output), '\n'.join(nav)


def build():
    labels = json.loads((ROOT / 'topic_labels.json').read_text())
    trends = read_csv('topic_by_round.csv')
    rows = []
    for rq in ['RQ1', 'RQ2', 'RQ3']:
        for topic, label in enumerate(labels[rq]['labels'], 1):
            matches = {int(r['round']): float(r['mean_weight']) for r in trends if r['corpus'] == rq and int(r['topic']) == topic}
            rows.append([rq, label, *[f'{matches[r]:.1%}' if r in matches else '题目不同' for r in range(1, 5)]])
    report = (ROOT / 'report.md').read_text(encoding='utf-8')
    report = replace_table(report, 'TOPIC_TABLE', md_table(['问题', '候选主题', '第一轮', '第二轮', '第三轮', '第四轮'], rows))
    candidates = read_csv('model_candidates.csv')
    topics = read_csv('topics.csv')
    rows = []
    for rq, choice in labels.items():
        row = next(r for r in candidates if r['corpus'] == rq and int(r['k']) == len(choice['labels']))
        topic = next(r for r in topics if r['corpus'] == rq)
        rows.append([rq.replace('RQ3_extended', '混合题目辅助模型'), row['k'], row['n'], row['vocab'], f"{float(row['npmi']):.3f}", f"{float(topic['bootstrap_topic_similarity_mean']):.3f}"])
    report = replace_table(report, 'DIAGNOSTIC_TABLE', md_table(['语料', '主题数', '回答数', '词项数', 'NPMI', '重抽样相似度'], rows))
    (ROOT / 'report.md').write_text(report, encoding='utf-8')
    body, nav = render_markdown(report)
    # Date is in the masthead; remove its duplicate first paragraph.
    body = re.sub(r'^<p>39名学生.*?</p>\n?', '', body, count=1)
    codes = read_csv('provisional_codebook.csv')
    code_rows = ''.join('<tr><td>' + '</td><td>'.join(inline(r[k]) for k in ['code', 'rq', 'label', 'inclusion', 'exclusion_or_boundary']) + '</td></tr>' for r in codes)
    downloads = ''.join(f'<a href="{name}" download>{label}<span>CSV ↓</span></a>' for name, label in [
        ('topics.csv', '主题汇总'), ('topic_by_round.csv', '分轮权重'),
        ('paired_changes.csv', '配对变化'), ('sensitivity.csv', '敏感性分析'),
        ('analysis_denominators.csv', '回答分母'), ('model_candidates.csv', '模型诊断'),
        ('provisional_codebook.csv', '候选代码本')])
    page = f'''<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 同伴评阅中的反馈、修订与学习</title>
<meta name="description" content="39名学生四轮反思的探索性分析：反馈类型、修订行动、学习收获，以及后续人工编码与理论解释。">
<meta name="color-scheme" content="light"><link rel="stylesheet" href="site.css"></head>
<body><a class="skip" href="#report">跳至报告正文</a>
<header class="masthead"><a class="wordmark" href="#">AI PEER REVIEW <span>/ RESEARCH NOTES</span></a><a href="{REPO}">GitHub 仓库 ↗</a></header>
<div class="layout"><aside aria-label="报告目录"><p class="toc-label">报告目录</p><nav>{nav}<a href="#downloads"><span>{len(nav.splitlines()) + 1:02}</span>下载与代码本</a></nav><p class="aside-note">探索性计算分析<br>2026年9月27日</p></aside>
<main id="report"><header class="report-head"><p class="eyebrow">反馈 · 修订 · 学习</p><h1>AI 同伴评阅中的<br>反馈、修订与学习</h1><p class="subtitle">39名学生四轮反思的探索性分析</p><div class="metadata"><span>39名参与者</span><span>427条有效回答</span><span>独立人工编码待完成</span></div></header>
<article>{body}</article>
<section id="downloads"><h2>下载与代码本</h2><p>下载汇总表，或在GitHub查看数据清理、建模和出图代码。表格中不含学生原话和个体编号。</p><div class="downloads">{downloads}</div>
<details><summary>查看36个候选代码及边界</summary><p class="note">代码由AI辅助起草，供开放编码讨论，尚未完成独立人工验证。</p><div class="table-scroll" role="region" tabindex="0" aria-label="候选代码本，可横向滚动"><table><thead><tr><th>代码</th><th>RQ</th><th>名称</th><th>纳入范围</th><th>边界</th></tr></thead><tbody>{code_rows}</tbody></table></div></details>
<p class="source-link"><a href="{REPO}">查看全部代码与复现说明 ↗</a> · <a href="report.md" download>下载报告 Markdown</a></p></section>
<footer>公开版本保留汇总结果与方法。学生原始资料和内部复核文件留在本地。<br>计算结果与候选代码用于支持研究者阅读，最终解释需经人工核查。</footer>
</main></div></body></html>'''
    (ROOT / 'index.html').write_text(page, encoding='utf-8')
    print('Built index.html from public aggregate results.')


if __name__ == '__main__':
    build()
