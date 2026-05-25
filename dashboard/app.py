"""
No-Slop Research Dashboard — Flask web interface for managing
adversarial research pipelines, viewing results, and configuring API keys.

Run: python -m dashboard.app
Port: 5060 (configurable via DASHBOARD_PORT env var)
"""

import os
import sys
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone
from functools import wraps

from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, session, g, send_from_directory)

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import ResearchPipeline, get_db, DB_PATH

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

# ===== Database Setup =====

def init_db():
    """Initialize all database tables."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            key_name TEXT NOT NULL,
            key_value TEXT NOT NULL,
            base_url TEXT,
            model_name TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dashboard_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            config TEXT DEFAULT '{}',
            status TEXT DEFAULT 'queued',
            run_id TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def get_conn():
    """Get a database connection for the current request."""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ===== API Key Management =====

PROVIDER_PRESETS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "key_prefix": "sk-"
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o", "google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
        "key_prefix": "sk-or-"
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4", "claude-3.5-haiku", "claude-3-opus"],
        "key_prefix": "sk-ant-"
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "key_prefix": "gsk_"
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "key_prefix": ""
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "key_prefix": "sk-"
    },
    "custom": {
        "name": "Custom (OpenAI-compatible)",
        "base_url": "",
        "models": [],
        "key_prefix": ""
    }
}


def mask_key(key: str) -> str:
    """Mask an API key for display."""
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ===== Routes =====

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


# --- API: Keys ---

@app.route("/api/keys", methods=["GET"])
def list_keys():
    conn = get_conn()
    keys = conn.execute("SELECT id, provider, key_name, key_value, base_url, model_name, is_active, created_at FROM api_keys ORDER BY created_at DESC").fetchall()
    result = []
    for k in keys:
        result.append({
            "id": k["id"],
            "provider": k["provider"],
            "key_name": k["key_name"],
            "key_masked": mask_key(k["key_value"]),
            "base_url": k["base_url"],
            "model_name": k["model_name"],
            "is_active": bool(k["is_active"]),
            "created_at": k["created_at"]
        })
    return jsonify(result)


@app.route("/api/keys", methods=["POST"])
def add_key():
    data = request.json
    provider = data.get("provider", "custom")
    key_name = data.get("key_name", f"{provider} key")
    key_value = data.get("key_value", "")
    base_url = data.get("base_url", "")
    model_name = data.get("model_name", "")

    if not key_value:
        return jsonify({"error": "key_value is required"}), 400

    # Auto-fill from presets
    if provider in PROVIDER_PRESETS and provider != "custom":
        preset = PROVIDER_PRESETS[provider]
        if not base_url:
            base_url = preset["base_url"]

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO api_keys (provider, key_name, key_value, base_url, model_name, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
        (provider, key_name, key_value, base_url, model_name, now, now)
    )
    conn.commit()
    return jsonify({"success": True, "message": f"API key added for {provider}"})


@app.route("/api/keys/<int:key_id>", methods=["DELETE"])
def delete_key(key_id):
    conn = get_conn()
    conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
    conn.commit()
    return jsonify({"success": True})


@app.route("/api/keys/<int:key_id>/toggle", methods=["POST"])
def toggle_key(key_id):
    conn = get_conn()
    conn.execute("UPDATE api_keys SET is_active = NOT is_active WHERE id = ?", (key_id,))
    conn.commit()
    return jsonify({"success": True})


@app.route("/api/providers", methods=["GET"])
def list_providers():
    return jsonify(PROVIDER_PRESETS)


# --- API: Research ---

@app.route("/api/research/start", methods=["POST"])
def start_research():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    config = data.get("config", {})
    max_rounds = config.get("max_rounds", int(os.environ.get("MAX_ADVERSARIAL_ROUNDS", "3")))

    # Get active API key
    conn = get_conn()
    active_key = conn.execute("SELECT * FROM api_keys WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1").fetchone()

    if active_key:
        config["api_key"] = active_key["key_value"]
        config["base_url"] = active_key["base_url"]
        config["model_name"] = active_key["model_name"]
        config["provider"] = active_key["provider"]

    # Create pipeline
    pipeline = ResearchPipeline(topic=topic, config=config)
    run_id = pipeline.run_id

    # Queue it
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO research_queue (topic, config, status, run_id, created_at) VALUES (?, ?, 'running', ?, ?)",
        (topic, json.dumps(config), run_id, now)
    )
    conn.commit()

    # Run pipeline (synchronous for now — could be async with Celery/threading)
    try:
        result = pipeline.run()
        # Update queue
        conn.execute("UPDATE research_queue SET status = 'completed' WHERE run_id = ?", (run_id,))
        conn.commit()
        return jsonify({"success": True, "run_id": run_id, "result": result})
    except Exception as e:
        conn.execute("UPDATE research_queue SET status = 'error' WHERE run_id = ?", (run_id,))
        conn.commit()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/research/runs", methods=["GET"])
def list_runs():
    conn = get_conn()
    runs = conn.execute("SELECT * FROM research_runs ORDER BY created_at DESC LIMIT 50").fetchall()
    return jsonify([dict(r) for r in runs])


@app.route("/api/research/runs/<run_id>", methods=["GET"])
def get_run(run_id):
    conn = get_conn()
    run = conn.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return jsonify({"error": "Run not found"}), 404

    phases = conn.execute("SELECT * FROM phase_results WHERE run_id = ? ORDER BY round_num, id", (run_id,)).fetchall()
    subagents = conn.execute("SELECT * FROM subagent_logs WHERE run_id = ? ORDER BY round_num, id", (run_id,)).fetchall()

    def safe_dict(row, truncate_large=True):
        """Convert Row to dict, sanitizing all text for safe JSON serialization."""
        d = dict(row)
        for k, v in d.items():
            if isinstance(v, str):
                # Remove null bytes and normalize line endings
                v = v.replace('\x00', '').replace('\r\n', '\n').replace('\r', '')
                # Escape any remaining control characters
                v = ''.join(ch if ord(ch) >= 32 or ch in '\n\t' else '' for ch in v)
                if truncate_large and len(v) > 2000:
                    v = v[:2000] + f"\n\n[... truncated, {len(v)} total chars]"
                d[k] = v
        return d

    return app.response_class(
        json.dumps({
            "run": safe_dict(run),
            "phases": [safe_dict(p) for p in phases],
            "subagents": [safe_dict(s) for s in subagents]
        }, ensure_ascii=False),
        mimetype='application/json'
    )


@app.route("/api/research/runs/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    conn = get_conn()
    conn.execute("DELETE FROM subagent_logs WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM phase_results WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM research_runs WHERE id = ?", (run_id,))
    conn.execute("DELETE FROM research_queue WHERE run_id = ?", (run_id,))
    conn.commit()
    return jsonify({"success": True})


# --- API: Settings ---

@app.route("/api/settings", methods=["GET"])
def get_settings():
    conn = get_conn()
    settings = conn.execute("SELECT * FROM dashboard_settings").fetchall()
    return jsonify({s["key"]: s["value"] for s in settings})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    for key, value in data.items():
        conn.execute(
            "INSERT OR REPLACE INTO dashboard_settings (key, value) VALUES (?, ?)",
            (key, str(value))
        )
    conn.commit()
    return jsonify({"success": True})


# --- API: Health ---

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "db": os.path.exists(DB_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# ===== Main =====

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("DASHBOARD_PORT", "5060"))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    print(f"\n  No-Slop Research Dashboard")
    print(f"  Running on http://{host}:{port}")
    print(f"  Database: {DB_PATH}\n")
    app.run(host=host, port=port, debug=True)
