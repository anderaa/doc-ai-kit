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

## Scanned pages

A page that looks scanned -- fewer than `extraction.transcription.min_chars_per_page`
characters of text layer, **and** images covering at least `min_image_coverage` of it -- is
rendered to an image and transcribed by Claude, and the transcription goes into the cached
text in that page's place. Both conditions, because most short pages are just short: a
signature block, a blank page, an exhibit cover. On CUAD's 510 contracts, 230 pages had
under 100 characters and 2 of them were scans. This is decided **page by
page**: a report with twenty typed pages and five scanned pages of accounts gets those five
read, rather than passing on a healthy-looking average with them blank.

It needs `extraction.transcription.model` set in `config.yaml`, e.g.
`anthropic/claude-sonnet-5`. There is no default, because a corpus of clean PDFs never needs
it. Until it is set, `extract` lists the documents with unread pages; set it and run
`extract` again, and only those documents are redone. Transcriptions are cached per page in
`data/transcripts/`, so nothing is paid for twice. A page costs roughly 4,000 input tokens,
depending on `max_image_px`.

`transcription.mode: all_pages` transcribes every page. Use it when PDFs carry a text layer
that is garbage -- left by a bad earlier OCR -- which the character count cannot detect.
Sample a few cached texts after extracting a new corpus to find out.

The manifest records, per document, `thin_pages`, `transcribed_pages` and `unread_pages`
(pages still without complete text: not transcribed, failed, or cut off). Those counts
reach two places later. In error analysis they explain a task that scores zero -- no prompt
recovers information that is not in the text. In production QA they route the document to
human review, because text read from an image can still be misread.

If documents still have unread pages after transcription, decide deliberately: look at the
pages, or accept that they will not be answerable and say so in the report.

## Truncation

Declared once, in `config.yaml`, and applied identically everywhere. If you change it, every
cached document is stale: re-extract with `--force`, and re-run anything that was measured
against the old text.

## Exit criteria

- `data/text/` has one file per PDF.
- `extraction_manifest.csv` written, and any flagged documents understood.

## Next

`sample-labels`, unless labels already exist.
