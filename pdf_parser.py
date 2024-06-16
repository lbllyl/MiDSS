from pdfminer.high_level import extract_text

text = extract_text("11782_camera.pdf")

with open("./text.txt", "w", encoding='utf-8') as f:
    f.write(text)
