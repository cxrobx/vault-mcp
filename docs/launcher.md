# The launcher — design, and why it is this small

`src/vault_mcp/launcher.py` plus the `j` Alfred workflow: type a description
("acme last meeting", "vault note on relationships"), get the file. The README
covers how to use and install it. This covers what was decided, what was tried
and dropped, and what would justify changing it. Built 2026-09-18.

## Where it came from

The prompt was a per-keystroke "predictive launcher" demo: fuzzy-filter a list
of apps and files to ~13 rows, send them with the typed text to a hosted
decision model on every keystroke, blend its probabilities with the fuzzy score.
The demo's best query was "the pdf I just downloaded" — a file found by its
facts, not its name.

Three things changed the shape before any code:

1. **The useful queries were about documents, not apps.** Apps, toggles and the
   user's own short keywords are already faster by name. What no tool answered
   was "acme last seo report", "most recent globex proposal", "the agent
   debugging guide in onyx", "md on jev usecases" — a scope, a topic, sometimes
   a recency word, across the vault, the HTML artifacts, client folders and the
   docs inside code repos.
2. **Those are whole phrases, not keystrokes.** Run on a typing pause. The
   sub-150 ms budget that justified a hosted decision model went away.
3. **The retriever already existed here.** Hybrid dense + BM25 over the vault.
   What was missing was coverage (documents outside the vault) and a front end
   that understands routing words.

So it is a module in this repo rather than a new one: one index, one reindex
path, and `search_vault` gains the wider corpus too. A second repo would have
meant a second index merged at query time — which is exactly what made the
first test noisy (half of every shortlist came from the wrong corpus).

## What a test on real queries showed

A scratch index of the new material, queried beside a copy of the live one,
five real phrases:

| Finding | Consequence |
|---|---|
| Four of five targets were already in the top three with plain hybrid retrieval. | No model is needed to *find* them. |
| Routing words poison the topic. "vault note on relationships" retrieved notes about vaults and dropped the target to #9; "relationships" alone put it at #1–4. "acme last seo report" → #7; "acme seo report" → #1. | Parse scope / recency / kind off the phrase **before** embedding it. Plain code, a word list. |
| The fifth target was not in the top 30 under any phrasing. It was indexed, and was #1 for its own words — the file calls itself an "oversight review deck"; the person calls it "the SEO report". | A vocabulary gap. No reranker can fix it: a judge never sees a row the retriever dropped. Only more evidence on the row can (see *Open*). |
| Inside one client's folder everything embeds alike — meeting notes, the status page and the actual proposal within ~0.05 cosine for "globex proposal". | "Newest of the best N matches" returns the newest meeting note. Recency needs a different pool (below). |

## The design

```
phrase ─► parse ─► topic + scope folders + recent? + suffix
                      │
                      ├─ not recent ─► hybrid search within scope ─► best chunk per file
                      └─ recent     ─► files in scope whose PATH carries a topic word
                                       ─► by date (filename date, else mtime)
                                       ─► padded with the plain topic order
```

- **Routing words are read from the leading and trailing runs only.** "pricing
  for last mile delivery" keeps its "last".
- **Scopes come from the index, not a list.** Any single-word folder name is a
  scope if some folder of that name holds 8+ files; then every folder of that
  name is in it (a client's 3-file proposal folder belongs to the client's
  scope). The threshold exists because a small folder's name is usually just
  the topic ("jev" is a two-file folder *and* what you are asking about).
  Nothing personal is hard-coded, which matters in a public repo.
- **A scope word stays in the topic.** Harmless inside the scope, and the
  unscoped retry (when the scope has nothing on the topic) still knows what was
  asked.
- **Recency pools by path, not by match strength** — the fourth finding above.
  What says a file *is* a proposal or a meeting is its name and its folders.
- **Dates: filename first** (`2026-09-01`, `09.01.26`), else mtime, computed at
  query time — no schema change. A git-log fallback was planned and dropped:
  on a working laptop mtime is honest enough, and the log will say if not.
- **No daemon.** A cold call — interpreter, 110 MB index load, query embedding,
  search — measured 0.28 s. The plan said "daemon only if over 400 ms".
- **Not an MCP tool, and nothing in the code is named `find`.** Agents keep four
  tools; one that wants this behaviour passes `folder=` to `search_vault`. Two
  tools called "find" and "search" is the confusion to avoid.

## The name leg (2026-09-19)

Typing `acme doss` put the client's long status page first and the file
actually called "Engagement Dossier" second. Two causes, both structural:

1. **Nothing matched a partial word.** `build_fts_query` quotes every term, and
   FTS5 has no prefix matching there, so `"doss"` matched **zero** rows — the
   lexical leg vanished and the search silently degraded to dense-only. A
   phrase is typed a character at a time, so this is the normal case, not an
   edge one. Measured: `doss` → 0 lexical hits, `dossier` → 50.
2. **Titles were not stored.** The index held chunks; a file's title only ever
   existed inside a chunk's embedding prefix. Nothing could match "the document
   *called* X" — and a status page that mentions the dossier will always beat
   the dossier on content, because the dossier's name is one line of the status
   page and the subject is the whole of it.

So `files` gained a `title` column (an HTML `<title>`, else the filename),
backfilled by reading each file's head once — 607 files in ~0.5 s, no
re-embedding — and the launcher gained a leg that ranks by what a file is
*called* rather than what it says:

- `index_tokens` splits a file's **own** words (title + filename) from its
  **folders'** words. A client folder's name is shared by everything in it, so
  it places a file; only its own name identifies it.
- `name_match` — every term must hit, as a prefix, and at least one in the
  file's own name. This is identification, and it orders ahead of content.
- `name_score` — graded, any term, 2 for own / 1 for folder. This is
  *gathering*, used by the recency pool: "seo report" must reach both the SEO
  baseline and the engagement report, and neither carries both words.

**Date still wins inside a recency phrase.** The first cut sorted the recency
pool by match strength first and a 2026-04 file titled "Acme Meeting Notes"
beat every newer file in `Meetings/`. "Last" is a question about time: among
the files the words fit, the newest is the answer, and strength only breaks a
tie on the same date.

## The judge that was not built

The original idea put a model after retrieval to pick the winner. Measured on
the hard case (30 newest rows of one client folder, titles + dates, "which is
the last SEO report?"):

| Judge | Result |
|---|---|
| Small subscription model via CLI | 13–16 s per call (stripping startup cut local CPU 8.8 s → 0.7 s and barely moved the wall clock, so the wait is model-side), **and it picked the wrong row**. |
| Any judge, in principle | The same wrong row. The titles do not contain the answer — "weak judgment = missing evidence". |
| Hosted decision model | Fast enough, but the rows are client document titles and the vendor publishes no deletion window. The owner had reverted an email-triage integration with the same vendor the same day on those grounds. Ruled out for client-scoped rows, which are the valuable ones. |

So version 1 ships without one, and **logs every run and every pick with its
rank** (`~/.local/state/vault-mcp/launcher.jsonl`). A judge earns its place only
where the evidence is already on the row and code still cannot rank it —
exclusions and comparisons ("the proposal, not the draft", "the one before the
pricing change"). None of the five real phrases was one.

### Reading the log, after a couple of weeks of use

- Pick rank mostly 1–3 → done. Leave it alone.
- Picks at rank 4–8 that share a shape (exclusions, comparisons) → that is the
  judge's test set. Try a local model first (~1 s, nothing leaves the machine).
- `query` events with no following `pick` → the target was not on the list.
  That is retrieval or evidence; a judge would not have helped.

## What is in the index, and what was taken out (2026-09-19)

607 documents: the vault (524), `~/clients` (48), the Onyx pages the vault does
not already give (20), and the proposal sources (15).

**Repo docs were in and came out the day after.** 484 rows — 347 in `docs/`,
116 top-level `.md` (84 of them `CLAUDE.md` / `README.md` / `CHANGELOG.md` /
`AGENTS.md`), 21 in `plans/` and `specs/` — which was 44% of the index for a
class the owner does not open. Four of the eight results in the first real
screenshot came from it. The redundancy settles it: his own "md on jev
usecases" returned the vault note #1, the repo's `docs/jev.md` #2 and the Onyx
page #3 — what is worth re-reading is already filed into the vault or
Artifacts, and the repo copy is the working copy. `scripts/link-repo-docs.sh`
and the note-mount machinery stay; putting them back is one config line and a
~3-minute reindex.

**Agent files stay indexed but off a typed list.** `CLAUDE.md`, `AGENTS.md` and
their kin are written for an agent and never opened by a person, and they are
long enough to touch every topic their project touches — with repo docs gone,
the client folders' own pair took #1 and #2 for "acme dossier". `AGENT_FILES`
in `launcher.py` drops them from the typed list unless the phrase names one
("acme claude"), while `search_vault` still returns them, because an agent
asking about an engagement's posture wants exactly that file.

What the owner actually opens is client deliverables — dossiers, decks, meeting
notes — which are vault HTML and meeting notes, all still indexed.

## Open

- **The vocabulary gap.** The fix identified is evidence from sent mail: a
  document is "the report" because it was sent to the client, and the email's
  subject line carries the person's words where the filename does not. Who,
  when and subject, attached to the row of every linked document. On hold by
  the owner's call, with email search generally. Unchecked: how hosted links in
  an email map back to source files.
- **Email as a corpus.** The mail client's local database already has a
  full-text index (35k messages, 4 ms per query), so keyword + date search over
  email is nearly free. Embedding it is the heavy version and should wait for a
  query that fails without it.
- **PDFs** are not indexed — no text to embed.
- **Downloads / Desktop are out, deliberately** (declined 2026-09-19). The
  demo's own query, "the pdf I just downloaded", does not work here and is not
  meant to: those rows have a name, an extension and an mtime and no content,
  so they are a second metadata-only corpus, and ~1,100 download filenames
  would outnumber the whole document index. A file you downloaded minutes ago
  is already a sorted folder or the launcher's own file search. The gap this
  tool fills is a *document* you cannot name.
- **Multi-word folder names** ("Reading Notes") cannot be scope words.
