# Recognize ≠ Refuse — front-end

Vite + React (plain JS/JSX) front-end for the "RECOGNIZE ≠ REFUSE" safety dashboard. Talks
to the FastAPI backend (`src/danger_tracker/api.py`) at `/api/*`.

## Develop

```sh
npm install
npm run dev          # proxies /api -> http://localhost:8000 (see vite.config.js)
VITE_MOCK=1 npm run dev   # runs entirely offline with canned data, no backend/model needed
```

## Build

```sh
npm run build         # outputs to dist/, served by FastAPI in production
```

`node_modules/` and `dist/` are gitignored — build on the target box; don't commit build
output.
