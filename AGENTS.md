# Repository Guidelines

## Project Structure & Module Organization
This repository is a Jekyll-based academic site built on `al-folio`. Core site configuration lives in `_config.yml`. Page content is organized by Jekyll collections: `_pages/` for top-level pages, `_news/` for announcements, `_projects/` for project entries, `_talks/` for talk listings, and `_bibliography/` for publications. Shared templates live in `_includes/` and `_layouts/`; custom plugins are in `_plugins/`; Sass sources are in `_sass/`. Static assets such as images, PDFs, fonts, JavaScript, and CSS belong under `assets/`. Build helpers live in `bin/`, and deployment/build automation is defined in `.github/workflows/`.

## Build, Test, and Development Commands
Use Docker for the most reliable local workflow:

```bash
docker compose up
```

This serves the site with live reload at `http://localhost:8080`.

For a local Ruby setup:

```bash
bundle install
bundle exec jekyll serve --lsi
bundle exec jekyll build --lsi
```

Use `bin/cibuild` for the CI-equivalent build and `bin/deploy` only when intentionally publishing a built site branch. If you change styles significantly, run `purgecss -c purgecss.config.js` after a production build, matching the GitHub Actions deploy workflow.

## Coding Style & Naming Conventions
Use 2-space indentation in YAML, Markdown front matter, and Liquid-heavy templates. Keep Markdown files concise and front matter explicit. Follow existing collection naming patterns such as `_news/announcement_1.md` and date-based posts like `YYYY-MM-DD-title.md` when adding blog posts. Prefer lowercase, hyphenated filenames for pages and assets. Run `pre-commit run --all-files` before opening a PR; the configured hooks check YAML, trailing whitespace, end-of-file newlines, and large files.

## Testing Guidelines
There is no dedicated unit test suite in this repository. Validation is build-based: run `bundle exec jekyll build --lsi` before submitting changes, then manually review the affected pages in `docker compose up`. For UI or content changes, verify navigation, generated collection pages, and any assets referenced from front matter.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Update news` and `Add new preprints and format papers.bib`. Keep commits focused and descriptive. PRs should include a brief summary, note any affected pages or collections, link the relevant issue when applicable, and attach screenshots for visible layout or styling changes. Avoid mixing unrelated content updates with structural or theme edits.
