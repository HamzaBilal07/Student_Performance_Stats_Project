# Deployment Guide

## GitHub

1. Create a new GitHub repository.
2. Push the full `student-performance-stats-project` folder.
3. Keep these files committed:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
   - `data/student_performance_factors.csv`
   - `assets/logo_placeholder.png`
   - `src/`

Generated files in `reports/` and `outputs/graphs/` are ignored by `.gitignore`.

## Recommended Live Deployment

For this exact Streamlit app, use a host that supports a long-running Python process:

- Streamlit Community Cloud
- Render
- Railway
- VPS/server hosting

## Vercel Note

Vercel is not the best direct target for a Streamlit app because Streamlit runs as a persistent Python web process and uses WebSocket-style communication. Vercel Python is designed for serverless HTTP functions, and Vercel Functions do not support acting as a WebSocket server.

Useful references:

- Vercel Python Runtime: https://vercel.com/docs/functions/runtimes/python
- Vercel WebSocket limits: https://vercel.com/docs/limits/overview
- Streamlit deployment docs: https://docs.streamlit.io/deploy

If Vercel is required, use Vercel for a separate frontend/landing page and host the Streamlit application on a Streamlit-compatible service.
