from flask import Flask, render_template, request
import main  # Import your orchestrator from main.py

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with something secure

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the raw content from a hidden form field named "hidden_input"
        raw_text = request.form.get("hidden_input", "")
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        if not lines:
            # No statements entered
            return render_template("index.html", content="", error="Please enter at least one statement.")

        # Run the full orchestrated process with results
        results = main.orchestrate_custom_process_with_results(lines)

        # Now we build a new HTML snippet with color highlights for each line
        # so that we can re-inject it into the contenteditable div.
        highlighted_content = []
        for item in results:
            statement = item["statement"]
            status = item["status"]
            # Determine background color based on status
            if status == "INCORRECT":
                # Subtle red
                line_html = f'<div style="background-color: #ffd6d6;">{statement}</div>'
            elif status == "PARTIALLY CORRECT":
                # Subtle orange
                line_html = f'<div style="background-color: #fff4d6;">{statement}</div>'
            else:
                # No highlight
                line_html = f'<div>{statement}</div>'
            highlighted_content.append(line_html)

        # Join into one string
        new_content = "".join(highlighted_content)

        return render_template(
            "index.html",
            content=new_content,
            error=None
        )
    else:
        return render_template("index.html", content="", error=None)

if __name__ == "__main__":
    app.run(debug=True)
