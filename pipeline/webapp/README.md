# Paracodex Pipeline Web Application

A modern web interface to run the **Paracodex code translation pipeline** — translate parallel code between different programming APIs (CUDA, OpenMP, OpenCL, HIP, SYCL, etc.) using AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

## Quick Start

### Prerequisites

- **Python 3.8+**
- The rest of the pipeline dependencies must also be set up (see the root-level `README.md` in this repo)

### Running Locally

```bash
# 1. Clone the repo (if you haven't already)
git clone <repo-url>
cd pipeline_refactored/webapp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
# OR use the convenience script:
bash start.sh
```

Then open **http://localhost:5000** in your browser.

> **Note:** `start.sh` will automatically install dependencies if they are missing.

---

## Usage

1. **Source Directory** — Pick the folder containing the code you want to translate (use the 📁 browser or type the path)
2. **Source API** — The parallel API currently used in your code (e.g. `cuda`)
3. **Target API** — The API you want to translate _to_ (e.g. `hip`)
4. **Model** _(optional)_ — Which AI model to use for translation
5. Click **Start Pipeline** and watch live logs stream in

### Supported API Pairs

| From | To |
|------|-----|
| serial | omp, cuda, ocl, acc, hip, sycl |
| omp | serial, cuda, ocl, acc, hip, sycl |
| cuda | hip, sycl, ocl, omp |
| … | … |

All combinations work as long as the underlying pipeline has skills for that pair.

---

## Project Structure

```
webapp/
├── app.py              # Flask server — REST API + job management
├── requirements.txt    # Python dependencies
├── start.sh            # Convenience startup script
├── .gitignore          # Excludes jobs.db, __pycache__, etc.
└── static/
    ├── index.html      # Main UI
    ├── css/styles.css  # Styling
    └── js/main.js      # Frontend logic
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Get defaults and available APIs |
| `/api/browse-directory?path=…` | GET | Browse server filesystem |
| `/api/start-pipeline` | POST | Launch a new job |
| `/api/active-jobs` | GET | List recent jobs |
| `/api/job-status/<id>` | GET | Get job status |
| `/api/job-progress/<id>` | GET | Get stage + artifacts |
| `/api/logs/<id>` | GET | Stream logs (SSE) |
| `/api/kill-job/<id>` | POST | Kill a running job |
| `/api/artifact/<id>/<path>` | GET | Read a job artifact |

## Troubleshooting

**Port 5000 already in use?**
```bash
# Use a different port:
FLASK_RUN_PORT=8080 python app.py
```
Or edit the last line of `app.py`: `app.run(..., port=8080, ...)`

**Database errors?**
```bash
rm webapp/jobs.db   # Delete stale DB — it's recreated on next start
python app.py
```

**Pipeline not found?**
Make sure you're running `python app.py` from inside the `webapp/` directory, or use `start.sh` which handles this automatically.
