import cv2
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image
from reportlab.lib.utils import ImageReader
# import pytesseract
from tqdm import tqdm
import numpy as np
from flask import Flask, request,jsonify, send_file
# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
# from werkzeug.utils import secure_filename
import os
# from wtforms.validators import InputRequired

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\HP\vscode_projects\vidtopdf\Code\API\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'

def extract_frames(video_path, frame_rate, threshold):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Set the desired frame rate (in frames per second)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
    fps =51
    print('Frames Per second', fps)

    print('Frames Extraction Started...')
    n = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if (frame_rate * n) % fps == 0:
            is_duplicate = False
            for existing_frame in frames:
                if is_similar(frame, existing_frame, threshold):
                    is_duplicate = True
                    break
            if not is_duplicate:
                frames.append(frame)
                i += 1
        n += 1
        if not ret:
            break

    cap.release()
    print('Frames Extraction Done!')
    print('Total Frames:', len(frames))
    return frames


def is_similar(frame1, frame2, threshold=0.9):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hist_frame1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
    hist_frame2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

    hist_frame1 /= hist_frame1.sum()
    hist_frame2 /= hist_frame2.sum()

    intersection = cv2.compareHist(hist_frame1, hist_frame2, cv2.HISTCMP_INTERSECT)
    return intersection >= threshold

# def construct_dict(frame):
#     word_coord = {}
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     _, frame = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
#     detections = pytesseract.image_to_data(frame)

#     for entry in detections.splitlines()[1:]:  # Skip the header line
#         data = entry.split('\t')

#         if len(data) == 12 and data[11] != '-1':  # Ensure it's a valid entry with non-empty text
#             word = data[11]
#             x, y, w, h = map(float,(data[6], data[7], data[8], data[9]))
#             coordinates = (x, y, x + w, y + h)
#             word_coord[word] = coordinates

#     return word_coord

# def similar_patch(word_coord1,word_coord2):
#     similar_dict = {}
#     list_k1 = list(word_coord1.keys())
#     list_k2 = list(word_coord2.keys())

#     for w1,k1 in word_coord2.items():
#         for w2,k2 in word_coord1.items():
#             # print(f'w1: {w1}, w2: {w2}')
#             if len(similar_dict) != 0:
#                 if w1 == w2 and (list_k2[list_k2.index(w1)-1]==list_k1[list_k1.index(w1)-1]):
#                     similar_dict[w1] = k1
#                     break
#                 else:
#                     continue
#             else:
#                 if w1 == w2:
#                     similar_dict[w1] = k1
#                     break
#                 else:
#                     continue
#     return similar_dict


# def ocr_masking(next_frame, similar_dict):
#     end_coord = list(similar_dict.items())[-1][1]
#     new_y_coord = int(end_coord[1])

#     (x1, y1, x2, y2) = (1, new_y_coord, 886 , 260)
#     next_frame[y2:y1,x1:x2,:] = 0

#     return next_frame


def frames_to_pdf(frames, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=(frames[0].shape[1], frames[0].shape[0]))

    print('Frames to PDF Started...')
    for idx, frame in enumerate(frames):
        img_buffer = BytesIO()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convert to PIL Image
        frame_pil.save(img_buffer, format='JPEG')

        img_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        img_reader = ImageReader(img_buffer)

        c.drawImage(img_reader, 0, 0, width=frame.shape[1], height=frame.shape[0])
        c.showPage()

    c.save()
    print('PDF with frames created successfully!')

# class UploadFileForm(FlaskForm):
#     file = FileField("File", validators=[InputRequired()])
#     submit = SubmitField("Upload File")


# @app.route('/', methods=['GET',"POST"])
# @app.route('/home', methods=['GET',"POST"])
# def home():
#     form = UploadFileForm()
#     return render_template('index.html', form=form)

    
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the 'file' key is in the request.files dictionary
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # Check if the file has a name
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
        file.save(full_filename)
    
        # with BytesIO() as video_buffer:
        #     video_buffer.write(file.read())
        #     video_data = video_buffer.getvalue()
        # print(type(video_data))
        # print(video_data.__sizeof__())
        #Process the video without saving it to disk
        output_pdf = process_video_logic(full_filename)

        # Send the generated PDF as a response
        return send_file(output_pdf, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500




def process_video_logic(video_file):
    frames = extract_frames(video_file, frame_rate=1, threshold=0.89)
    # print(len(frames))
    # masked_frames = []
    # for i in tqdm(range(len(frames))):
    #     if i == 0:
    #         masked_frames.append(frames[i])
    #     else:
    #         coord1 = construct_dict(frames[i-1])
    #         coord2 = construct_dict(frames[i])
    #         similar_dict = similar_patch(word_coord1=coord1, word_coord2=coord2)
    #         if len(similar_dict) > 10:
    #             masked_frame = ocr_masking(frames[i], similar_dict)
    #             masked_frames.append(masked_frame)
    #         else:
    #             masked_frames.append(frames[i])

    output_pdf = 'Output_vid1_mod.pdf'
    frames_to_pdf(frames, output_pdf)

    return output_pdf

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
