import os
import sys
import threading
from flask import Flask, request, render_template, redirect, url_for, session

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.insert(0, BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT,  "frontend", "static"),
)

app.secret_key    = "super_secret_key_change_me_2026"
VALID_USERNAME    = "varshu"
VALID_PASSWORD    = "123456"

detection_thread  = None

# ── routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session["logged_in"] = True
            session["username"]  = username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    import detection
    return render_template("dashboard.html",
                           detection_running=detection.detection_running)


@app.route("/start_detection", methods=["POST"])
def start_detection_route():
    global detection_thread
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    import detection
    if detection_thread is None or not detection_thread.is_alive():
        detection.detection_running = True
        detection_thread = threading.Thread(target=detection.start_detection,
                                            daemon=True)
        detection_thread.start()
        print("Detection thread started")
    else:
        print("Detection already running")

    return redirect(url_for("dashboard"))


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    import detection
    detection.detection_running = False
    print("Detection stop requested")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    try:
        import detection
        detection.detection_running = False
    except Exception:
        pass
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    print("Flask backend starting...")
    app.run(host="0.0.0.0", port=5000, debug=False)