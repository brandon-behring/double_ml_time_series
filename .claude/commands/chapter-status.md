# /chapter-status

Show current chapter progress and word counts.

## Actions

1. Count words in each chapter:
```bash
for f in chapters/chapter_*.tex; do
  echo "$f: $(detex "$f" 2>/dev/null | wc -w) words"
done
```

2. Show PDF page count:
```bash
pdfinfo main.pdf 2>/dev/null | grep Pages || echo "PDF not built"
```

3. Read ROADMAP.md for phase status:
```bash
head -30 ROADMAP.md
```

4. Show recent git activity:
```bash
git log --oneline -5
```

## Expected Output

```
Chapter Status:
- Chapter 1: XXXX words (Complete)
- Chapter 2: XXXX words (Complete)
- Chapter 3: XXXX words (In Progress)

PDF: YY pages
Current Phase: 1B

Recent Activity:
[last 5 commits]
```
