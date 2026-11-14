import markdown
from xhtml2pdf import pisa
import os
import re
import shutil
from pathlib import Path

def convert_md_to_pdf(input_md, output_pdf, artifact_dir):
    # 0. Copy images from artifact dir to current dir
    image_files = [
        "preprocessing_pipeline.png",
        "system_architecture.png",
        "resnet50_architecture.png",
        "lstm_architecture.png",
        "confusion_matrix.png"
    ]
    
    project_dir = os.getcwd()
    print(f"Current Working Directory: {project_dir}")
    
    print("Checking and copying images...")
    for img_name in image_files:
        artifact_img_path = os.path.join(artifact_dir, img_name)
        local_img_path = os.path.join(project_dir, img_name)
        
        if os.path.exists(artifact_img_path):
            shutil.copy2(artifact_img_path, local_img_path)
        else:
            print(f"Warning: Image not found in artifacts: {img_name}")

    # 1. Read Markdown
    with open(input_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Preprocess Paths
    # Use simple local filenames. xhtml2pdf should resolve them relative to CWD.
    
    def replace_path(match):
        filename = match.group(1) # e.g. "foo.png"
        return f"({filename})"

    # Regex to capture the filename
    text = re.sub(r'\(file:///.*?/([^/]+\.png)\)', replace_path, text)
    
    # 3. Convert to HTML
    html_text = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Add CSS
    full_html = f"""
    <html>
    <head>
    <style>
        @page {{ size: a4 portrait; margin: 2cm; }}
        body {{ font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.5; }}
        h1 {{ font-size: 18pt; text-align: center; color: black; font-weight: bold; margin-bottom: 20px; }}
        h2 {{ font-size: 14pt; border-bottom: 1px solid black; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }}
        h3 {{ font-size: 12pt; margin-top: 15px; font-weight: bold; }}
        img {{ width: 100%; max-width: 600px; height: auto; display: block; margin: 15px auto; }}
        code {{ font-family: monospace; background-color: #f0f0f0; padding: 2px; }}
        pre {{ background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap; font-size: 9pt; }}
    </style>
    </head>
    <body>
    {html_text}
    </body>
    </html>
    """

    # 4. Write PDF
    # IMPORTANT: We can't easily set base_url for CreatePDF in this version maybe?
    # But usually it works.
    
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            src=full_html,
            dest=result_file,
            encoding='utf-8'
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created PDF: {os.path.abspath(output_pdf)}")

if __name__ == "__main__":
    artifact_dir = r"C:\Users\viswa\.gemini\antigravity\brain\80762198-d8e3-4cff-9370-8051be0f4064"
    artifact_md = os.path.join(artifact_dir, "project_report.md")
    local_md = "project_report.md"
    output_path = "project_report.pdf"

    # Copy MD
    if os.path.exists(artifact_md):
        with open(artifact_md, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(local_md, 'w', encoding='utf-8') as f:
            f.write(content)
            
    convert_md_to_pdf(local_md, output_path, artifact_dir)
