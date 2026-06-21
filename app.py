import csv
import io
import json

from flask import Flask, request, render_template, jsonify, send_file

from engine import extract_text_from_pdf, rank_resumes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

# Simple in-memory cache of the last ranking result, used for CSV export.
# Fine for a single-user portfolio demo; not meant for concurrent production use.
LAST_RESULT = {"job_description": "", "rankings": []}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rank", methods=["POST"])
def rank():
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"success": False, "error": "Job description is required."}), 400

    resumes = []

    # 1) Pasted text resumes (textarea fields named "resumes")
    pasted = [t for t in request.form.getlist("resumes") if t and t.strip()]
    for i, text in enumerate(pasted, start=1):
        resumes.append({"filename": f"Pasted Resume {i}", "raw_text": text})

    # 2) Uploaded PDF resumes (file input named "resume_files")
    files = request.files.getlist("resume_files")
    for f in files:
        if f and f.filename.lower().endswith(".pdf"):
            try:
                text = extract_text_from_pdf(f.stream)
                resumes.append({"filename": f.filename, "raw_text": text})
            except Exception as e:
                resumes.append({"filename": f.filename, "raw_text": ""})

    if not resumes:
        return jsonify({"success": False, "error": "Please provide at least one resume (pasted text or PDF)."}), 400

    rankings = rank_resumes(job_description, resumes)

    LAST_RESULT["job_description"] = job_description
    LAST_RESULT["rankings"] = rankings

    return jsonify({"success": True, "rankings": rankings})


@app.route("/download-report")
def download_report():
    rankings = LAST_RESULT["rankings"]
    if not rankings:
        return jsonify({"error": "No ranking results to export yet. Run a ranking first."}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Candidate / Filename", "Match Score (%)", "Matched Skills", "Missing Skills"])
    for i, r in enumerate(rankings, start=1):
        writer.writerow([
            i,
            r["filename"],
            r["score"],
            ", ".join(r["matched_skills"]),
            ", ".join(r["missing_skills"]),
        ])

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="resume_ranking_report.csv",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
