import json
import os
import re


def escape_latex(text):
    # Basic LaTeX escaping that works safely in text mode
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def generate_latex_from_json(folder_path):
    latex_sections = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json") and filename.lower().startswith("task"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

            match = re.match(r"Task(\d+)_([a-zA-Z0-9_]+)\.json", filename)
            if match:
                task_number = match.group(1)
                raw_name = match.group(2)
                task_name = raw_name.replace("_", " ").title()
                subsection_title = f"\\subsection{{Task{task_number}: {task_name}}}"

                description = data.get("Description", "").strip()
                escaped_description = escape_latex(description)

                latex_section = f"""{subsection_title}

\\begin{{quote}}
{escaped_description}
\\end{{quote}}
"""
                latex_sections.append(latex_section)

    return "\n".join(latex_sections)


# Example usage in current directory
folder = os.path.dirname(os.path.abspath(__file__))
latex_output = generate_latex_from_json(folder)
print(latex_output)
