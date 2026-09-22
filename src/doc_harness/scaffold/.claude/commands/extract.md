---
description: Turn the PDFs into the cached text everything else reads
---

# extract

## Entry conditions

- PDFs in `data/pdfs/`.
- The truncation policy decided in `config.yaml`.

## Why it exists

Text is extracted once and cached. Validation and production read the same cached text,
because metrics computed on one rendering of a document do not transfer to another: a
different extractor, or a different truncation length, is a different input and the accuracy
you measured no longer describes what you shipped.

Extraction is skipped for documents already cached, so re-running is cheap and safe.

## Steps

```
doc-harness extract
```

Writes `data/text/{doc_id}.md` and `data/extraction_manifest.csv`.

## The OCR flag

When a document's text layer falls below `extraction.ocr_chars_per_page` characters per
page, it is treated as missing: OCR runs if enabled, and either way the document is flagged
in the manifest.

That flag has to reach two places later. In error analysis it explains a task that scores
zero on those documents -- no prompt recovers information that is not in the text. In
production QA it routes the document to human review.

If the command reports flagged documents and OCR is disabled, decide deliberately: enable
the `ocr` extra, or accept that those documents will not be answerable and say so in the
report.

## Truncation

Declared once, in `config.yaml`, and applied identically everywhere. If you change it, every
cached document is stale: re-extract with `--force`, and re-run anything that was measured
against the old text.

## Exit criteria

- `data/text/` has one file per PDF.
- `extraction_manifest.csv` written, and any flagged documents understood.

## Next

`sample-labels`, unless labels already exist.
