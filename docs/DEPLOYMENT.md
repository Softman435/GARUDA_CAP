# Deployment notes (hosting for other people)

The local version has **no login**. Do not expose it to the internet as is. Before hosting you must add:

1. **Authentication and ownership** (not implemented): add a `users` table, a `user_id` on `projects`, and check it in `get_project()`
   and the download route in `backend/app/main.py`. Use Argon2id (`argon2-cffi`) for passwords and short-lived sessions or tokens;
   for Google sign-in use a maintained OpenID Connect library (e.g. Authlib) and validate state, nonce, issuer, audience and expiry.
   Required settings would be `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `APP_SECRET_KEY`, and e-mail settings only if you add password reset.
2. **Migrations**: add Alembic (`alembic init`) and generate a baseline from `backend/app/db.py`.
3. **HTTPS** at a reverse proxy (Caddy / Nginx), secure cookies, and `ALLOWED_ORIGINS` set to your real site.
4. **Production mode**: `APP_ENV=production` makes the server refuse to start without a 32+ character `APP_SECRET_KEY` and explicit origins.
5. **Database**: set `DATABASE_URL` to PostgreSQL (`pip install psycopg[binary]`). Keep videos on disk or object storage, never in the database.
6. **Storage**: videos live under `GARUDA_DATA_DIR/projects/<id>/`. For cloud storage add an adapter around the paths in `jobs.py`.
7. **Hosting requirements**: a server (not a static host) with FFmpeg, enough RAM/CPU for the speech model, disk space, long request timeouts and large upload limits. Static hosts (e.g. Netlify/Vercel-style) can serve only `frontend/dist`; the Python backend must run elsewhere.
8. **Operations**: backups of the database and data folder, log monitoring, `pip-audit` / `npm audit`.

If a secret is ever exposed: remove it from the code/history, generate a new one, update the server environment, restart, and revoke the old one at its provider (for Google: delete the old client secret in Google Cloud Console → Credentials).

## B-roll later
Add a `clips` table (project, file, start, end, position) and extend `render.py` to build a filter graph that overlays clips while mapping audio from the original narration (`-map 0:a`). Not built.
