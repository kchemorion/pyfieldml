# JOSS paper source

This directory holds the Journal of Open Source Software (JOSS) submission
for `pyfieldml`.

- `paper.md` — the paper itself, in JOSS's prescribed format.
- `paper.bib` — BibTeX references.
- `figures/` — any figures referenced from `paper.md`.

## Rendering the paper locally

JOSS provides a Docker-based builder called `openjournals/inara`:

```bash
docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara paper.md
```

Output: `paper.pdf` + `paper.jats` alongside `paper.md`.

## Submission

Once the paper is final, submit at
<https://joss.theoj.org/papers/new> with:

- Repository URL: `https://github.com/kchemorion/pyfieldml`
- Branch: `main`
- Version: the tagged release (e.g. `v1.0.0`)
- Zenodo archive DOI: obtained after a Zenodo release is linked to the
  GitHub release.

## Reviewer responses

If and when the JOSS review process flags items, record the exchange in
`review_responses.md` under this directory.
