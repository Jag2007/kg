# Deploying Knowledge Graph Studio

## Short answer: one service is enough

This app is **one FastAPI service that serves both the website and the API**. The UI in
`static/` is plain HTML/CSS/JS, and the same server returns it at `/`. Deploy it once on Render
and you get a single URL that works end to end.

Splitting into a frontend (Vercel) and a backend (Render) is **optional**. It's supported (see
Option B), but it adds CORS settings and a second deploy and doesn't make anything faster. Use
Option A unless you specifically want the frontend on Vercel.

| | Option A: one Render service | Option B: Vercel + Render |
| --- | --- | --- |
| Deploys to manage | 1 | 2 |
| Extra config | none | `config.js` API URL + `KG_CORS_ORIGINS` |
| Cost | free | free |
| **Recommended** | ✅ | only if you need Vercel |

> Your existing Hebbrix app (`knowledge-graphs-rbm9.onrender.com`, repo `knowledge_graphs`) is
> separate. Deploying `kg` creates a **new** service and doesn't touch the old one.

---

## Before you start

1. The code is on GitHub at **github.com/Jag2007/kg** (already pushed).
2. Log in to **render.com** with the **GitHub account Jag2007** ("Sign in with GitHub"). This
   lets Render see the `kg` repo. If Render asks for repository access, allow `Jag2007/kg`.

---

## Option A: one Render web service (recommended)

### A1. Create the service from the blueprint (easiest)

The repo contains `render.yaml`, which already holds every setting.

1. In Render, click **New +** → **Blueprint**.
2. Pick the repository **Jag2007/kg** and click **Connect**.
3. Render reads `render.yaml` and shows a service called **kg-studio**. Click **Apply**.
4. It may ask for the values marked `sync: false` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`,
   `OPENAI_ANSWER_MODEL`). **Leave them empty for now** and continue.
5. Wait for the first build, about 3–6 minutes. Watch the **Logs** tab; it ends with
   `Uvicorn running on http://0.0.0.0:10000`.
6. Open the URL Render shows (e.g. `https://kg-studio.onrender.com`).
7. Check `https://<your-url>/health`. You should see `"graph_backend": "embedded"` and
   `"spacy_ready": true`.
8. Upload a PDF and ask a question. Done.

### A1 (alternative). Create the service by hand

If you prefer not to use a blueprint: **New +** → **Web Service** → choose **Jag2007/kg**, then:

| Field | Value |
| --- | --- |
| Runtime | Python 3 |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Instance type | Free |
| Health check path | `/health` |

Environment variables (**Environment** tab → **Add Environment Variable**):

| Key | Value |
| --- | --- |
| `PYTHON_VERSION` | `3.12.10` |
| `KG_GRAPH_BACKEND` | `embedded` |
| `KG_EXTRACTOR` | `spacy` |
| `KG_EMBEDDING_BACKEND` | `local` |
| `KG_ANSWER_LLM` | `none` |

Click **Create Web Service** and continue from step 5 above.

### A2. Better answers on Render (optional, recommended)

Render can't run Ollama, so by default the deployed app uses **rule-based / graph-chain
answers**. They work, but they're less fluent than the local model. To have a model write the
answers, connect any **OpenAI-compatible** API. In the service → **Environment**, set:

| Key | Example value |
| --- | --- |
| `KG_ANSWER_LLM` | `openai` |
| `OPENAI_API_KEY` | your key |
| `OPENAI_BASE_URL` | the provider's base URL (see below) |
| `OPENAI_ANSWER_MODEL` | a model name from that provider |

Provider examples:
- **Hebbrix** (your key already works): base URL `https://api.hebbrix.com/v1`, model `gpt-5-nano`
- **OpenAI**: base URL `https://api.openai.com/v1`, model `gpt-4o-mini`
- **Groq** (has a free tier): base URL `https://api.groq.com/openai/v1`, model `llama-3.1-8b-instant`

Click **Save Changes**. Render redeploys automatically. Upload a PDF again after the redeploy.

Extraction and the graph stay local (spaCy + graph store). The API is only used to phrase the
final answer.

### A3. Keep the graph between restarts (optional)

On Render's free plan the disk is temporary: the graph is wiped on every redeploy and restart,
so just upload the PDF again. To keep it, use a hosted Neo4j:

1. Create a free instance at **console.neo4j.io** (AuraDB **Free**, not the Pro trial). Save
   the connection URI and password it shows once.
2. In Render → **Environment**, set:
   - `KG_GRAPH_BACKEND` = `auto`
   - `NEO4J_URI` = `neo4j+s://xxxxxxxx.databases.neo4j.io`
   - `NEO4J_USER` = `neo4j`
   - `NEO4J_PASSWORD` = the password
3. Save. `/health` should now show `"graph_backend": "neo4j"`.

If Neo4j is unreachable, the app falls back to the embedded store automatically, so it never
goes down because of the database.

### A4. Things to know about the free plan

- The service **sleeps after ~15 minutes without traffic**. The first visit after that takes
  about 30–60 seconds to wake up, which is normal.
- 512 MB RAM is enough. The app uses about 160–200 MB.
- Every `git push` to `main` redeploys automatically (Settings → Auto-Deploy).

---

## Option B: frontend on Vercel + backend on Render

Only if you want the UI on Vercel.

### B1. Backend on Render

1. Do **Option A** fully. Note the backend URL, e.g. `https://kg-studio.onrender.com`.

### B2. Point the frontend at the backend

2. In the repo, edit `static/config.js`:
   ```js
   window.KG_API_BASE = "https://kg-studio.onrender.com";
   ```
3. Commit and push.

### B3. Frontend on Vercel

4. On **vercel.com** → **Add New…** → **Project** → import **Jag2007/kg**.
5. Set **Root Directory** to `static`, **Framework Preset** to **Other**, and leave the build
   command empty.
6. Click **Deploy**. You get a URL like `https://kg-xxxx.vercel.app`.

### B4. Allow the frontend to call the backend (CORS)

7. In Render → backend service → **Environment**, add:
   - `KG_CORS_ORIGINS` = `https://kg-xxxx.vercel.app` (comma-separate several URLs if needed)
8. Save and wait for the redeploy.
9. Open the Vercel URL. The status pills should turn green. If they say "offline", the browser
   console will usually show a CORS error. Re-check step 7 (exact URL, no trailing slash).

---

## Option C: Docker (any server or your laptop)

```bash
docker compose up --build        # Neo4j + app on http://localhost:8000
```

Ollama, if you want model-written answers, runs on the host: `ollama pull qwen2.5:3b`.

---

## Running locally (for reference)

```bash
cd "~/Desktop/KG FIX/kg"
docker start kg-neo4j            # or: docker compose up -d neo4j
.venv/bin/uvicorn app:app --reload
# open http://127.0.0.1:8000
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails at `en_core_web_sm` | Make sure `PYTHON_VERSION` is `3.12.10` and the build command is `pip install -r requirements.txt` |
| `/health` shows `spacy_ready: false` | The spaCy model didn't install. Use **Clear build cache & deploy** in Render |
| Page loads but "Graph: offline" | The backend is waking up (wait ~1 min), or a CORS problem in Option B |
| "Please upload a PDF before asking" after a while | The free instance restarted and wiped the temporary disk. Upload again, or use A3 |
| Answers are short or robotic on Render | Expected without a model. Set up A2 |
| Upload of a very large PDF is slow | spaCy processes every chunk. 50–100 pages take tens of seconds on the free CPU |
