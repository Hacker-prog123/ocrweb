from flask import Flask, render_template, request
from paddleocr import PaddleOCR
import os
import re
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__,
            template_folder=os.path.join(os.getcwd(), "template"),
            static_folder=os.path.join(os.getcwd(), "static"))

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr = PaddleOCR(
    det_model_dir='Inference/en_PP-OCRv3_det_infer',
    rec_model_dir='Inference/en_PP-OCRv4_rec_infer',
    cls_model_dir='Inference/ch_ppocr_mobile_v2.0_cls_infer',
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    use_dilation=True,
    det_db_box_thresh=0.2
)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(image)
    rgb_image = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_image)

def extract_so_fields_layout(ocr_result):
    target_labels = {
        "ESN": None,
        "MODULE SN": None,
        "PART NUMBER": [],
        "SN": []
    }
    exclusion_terms_part_number = {"ADDITIONAL REPAIR", "IN-HOUSE", "NA"}
    boxes = []
    for line in ocr_result:
        box, (text, conf) = line
        x1, y1 = box[0]
        x2, y2 = box[2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        boxes.append({'text': text.strip(), 'conf': conf, 'center': (cx, cy), 'box': box})

    for label in target_labels:
        if label == "SN":
            label_box = next((b for b in boxes if "SN" in b['text'].upper()
                              and all(x not in b['text'].upper() for x in ["MODULE", "ESN", "CSN", "TSN"])), None)
        else:
            label_box = next((b for b in boxes if b['text'].strip().upper() == label), None)

        if label_box:
            label_cx, label_cy = label_box['center']
            collected = []
            for b in boxes:
                if b == label_box:
                    continue
                b_cx, b_cy = b['center']
                dx = b_cx - label_cx
                dy = b_cy - label_cy

                if label == "SN":
                    if 50 < dx < 250 and 0 < dy < 300:
                        collected.append(b['text'])
                elif label == "PART NUMBER":
                    if abs(dx) < 100 and 0 < dy < 250:
                        if b['text'].strip().upper() not in exclusion_terms_part_number:
                            collected.append(b['text'])
                else:
                    if abs(dx) < 100 and 5 < dy < 60:
                        target_labels[label] = b['text']
            if label in ["SN", "PART NUMBER"]:
                target_labels[label] = collected or ["(Not found)"]
        else:
            target_labels[label] = "(Not found)"
    return target_labels

def extract_tv_fields_labels(ocr_result):
    label_map = {
        'ESN (CUSTOMER)': 'ESN',
        'MODULE S/N': 'MODULE SN',
        'PART NUMBER': 'PART NUMBER',
        'SERIAL NUMBER': 'SN'
    }
    tv_fields = {}
    for i, (box, (text, conf)) in enumerate(ocr_result):
        for tv_label, key in label_map.items():
            if tv_label in text.upper():
                if i + 1 < len(ocr_result):
                    value = ocr_result[i + 1][1][0].strip()
                    if key == 'ESN':
                        value = value.split('(')[0].strip()
                    tv_fields[key] = value
    return tv_fields

def clean_val(v):
    if isinstance(v, list):
        v = v[0] if v else ''
    return re.sub(r'[;()\s]', '', v.strip().upper())

@app.route('/', methods=['GET', 'POST'])
def index():
    mode = request.form.get('mode') or request.args.get('mode')
    result_text = ""
    extracted_fields = {}

    if request.method == 'POST':
        if mode == 'compare':
            so_file = request.files['so_image']
            tv_file = request.files['tv_image']

            so_path = os.path.join(app.config['UPLOAD_FOLDER'], "so_temp.jpg")
            tv_path = os.path.join(app.config['UPLOAD_FOLDER'], "tv_temp.jpg")

            so_bytes = so_file.read()
            tv_bytes = tv_file.read()

            with open(so_path, 'wb') as f:
                f.write(so_bytes)
            with open(tv_path, 'wb') as f:
                f.write(tv_bytes)

            so_result = ocr.ocr(so_path, cls=True)[0]
            tv_result = ocr.ocr(tv_path, cls=True)[0]

            so_fields = extract_so_fields_layout(so_result)
            tv_fields = extract_tv_fields_labels(tv_result)

            comparison = ""
            all_match = True
            for key in ['ESN', 'MODULE SN', 'PART NUMBER', 'SN']:
                so_val = clean_val(so_fields.get(key, ''))
                tv_val = clean_val(tv_fields.get(key, ''))
                if so_val == tv_val:
                    comparison += f"✅ {key} MATCHES — {so_val}\n"
                else:
                    comparison += f"❌ {key} MISMATCH — SO: '{so_val}' vs TV: '{tv_val}'\n"
                    all_match = False

            result_text = comparison + ("\n🎉 All fields match!" if all_match else "\n⚠️ Some fields do not match.")
            extracted_fields = {'SO': so_fields, 'TV': tv_fields}

        elif mode == 'part_vs_tv':
            part_file = request.files['part_image']
            part_path = os.path.join(app.config['UPLOAD_FOLDER'], "part_temp.jpg")

            part_bytes = part_file.read()
            with open(part_path, 'wb') as f:
                f.write(part_bytes)

            image = preprocess_image(part_path)
            part_result = ocr.ocr(np.array(image), cls=True)[0]

            extracted_lines = [f"{line[1][0]} ({line[1][1]*100:.2f}%)" for line in part_result]
            result_text = "\n".join(extracted_lines)
            extracted_fields = {"PART TEXT": extracted_lines}

    return render_template("index.html", result=result_text, fields=extracted_fields, mode=mode)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
