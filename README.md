# Hidden Signals — Northfield & Co.

An interactive Streamlit dashboard on Northfield & Co.'s daily e-commerce revenue and
marketing spend, built to surface the **hidden calendar factors** (day-of-week, the
BFCM window, a coordinated media-budget rhythm) that confound the raw correlation
between paid media spend and revenue.

Four tabs: **Overview** (trend + seasonality), **The Suspects** (raw correlation
ranking), **The Reveal** (partial correlation, PCA hidden factor, Simpson's-paradox
scatter, lag structure, rolling correlation), and **The Verdict** (event-study
uplifts, the six hidden factors mapped, and recommendations) — plus a raw **Data
Explorer** tab.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy it (Streamlit Community Cloud — free, no server to manage)

1. Push this folder to a GitHub repository (keep the folder structure as-is —
   `app.py`, `requirements.txt`, `data/Northfield_Co_Case_Study.xlsx`,
   `.streamlit/config.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repository/branch, and set **Main file path** to
   `app.py`.
4. Click **Deploy**. No secrets or environment variables are needed — the workbook
   ships inside the repo and is read with a path relative to `app.py`, so it works
   unchanged on Streamlit Cloud, Hugging Face Spaces, Render, or any other host.

## Deploy elsewhere

Any host that can run `pip install -r requirements.txt && streamlit run app.py`
works (Render, Railway, Fly.io, a Docker container, an EC2 box behind nginx). The
app has no external network calls and no secrets — the only input is the bundled
`.xlsx`.

Minimal Dockerfile if you want a container:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Project structure

```
.
├── app.py                                # the dashboard
├── requirements.txt
├── .streamlit/config.toml                # theme (matches the accompanying HTML report)
├── data/Northfield_Co_Case_Study.xlsx    # source workbook (Master data + Glossary tabs used)
└── README.md
```

## Notes on the analysis

- All relationships shown are **observational associations**, not proven causal
  effects — see the methodology note at the bottom of the app.
- The hidden-factor analysis (partial correlation, PCA, event study, lag structure)
  always runs on the **full 852-day history** for statistical reliability, even
  when the sidebar date filter narrows the Overview tab's charts.
- Swap in a different workbook by replacing `data/Northfield_Co_Case_Study.xlsx`
  with a same-shaped file (same column names on the `Master data` tab), or point
  `DATA_PATH` in `app.py` at a different location / add a file-uploader.
