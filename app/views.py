from flask import request, render_template, redirect, url_for, flash, jsonify, session
from werkzeug import secure_filename
import json, codecs
from app.utils import *
import numpy as np
import app.CONFIG as CONFIG
from app import app


@app.route('/')
def index():
    return render_template('index.html', status=session)


@app.route('/add_image', methods=['GET', 'POST'])
def add_image():
    if request.method == 'POST':
        known_faces = {}
        image = request.files['known_face_img']
        name = request.form['known_face_name']

        # check if the file exists then load the data
        if os.path.exists(CONFIG.DB_JSON_PATH):
            with open(CONFIG.DB_JSON_PATH) as file:
                known_faces = json.load(file)

        # if a face with the similar name already exists, notify the user
        if name in known_faces.keys():
            flash('The name you are adding is already in the database. Try another name !')
            session['status'] = 'failed'
            return redirect(url_for('index'))

        # change the name of the image before saving into the specified directory
        image_name = name + os.path.splitext(image.filename)[1]
        image.save(os.path.join(CONFIG.KNOWN_FACES_PATH, secure_filename(image_name)))

        # run the model on the new image
        face_encodings = learn(os.path.join(CONFIG.KNOWN_FACES_PATH, image_name))

        # saving the face encodings into the json file
        face_encodings_list = face_encodings.tolist()
        known_faces[name] = {}
        known_faces[name]['image'] = os.path.join(CONFIG.KNOWN_FACES_PATH, image_name)
        known_faces[name]['learn'] = face_encodings_list

        # np array into json serialization
        json.dump(known_faces, codecs.open(CONFIG.DB_JSON_PATH, 'w', encoding='utf-8'), separators=(',', ':'),
                  sort_keys=True, indent=4)

        # feedback message
        session['status'] = 'success'
        flash('Face was added to the database successfully !')
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/recognize_image', methods=["GET", "POST"])
def recognize_image():
    if request.method == 'POST':

        # get image data from the request and save into the specified directory
        image = request.files['rec_face_img']
        image.save(os.path.join(CONFIG.TEST_PATH, secure_filename(image.filename)))

        # load the faces data from the json file in the database
        text = codecs.open(CONFIG.DB_JSON_PATH, 'r', encoding='utf-8').read()
        known_faces_json = json.loads(text)
        known_people = []
        known_face_encodings = []

        # if no face is stored in the database then return
        if not known_faces_json.keys():
            session['status'] = 'failed'
            flash('There is no face stored in the database. Please add faces first and try again !')
            return redirect(url_for('index'))

        # creating two lists as required by the model
        for key in known_faces_json.keys():
            known_people.append(key)
            known_face_encodings.append(known_faces_json[key]['learn'])

        # running the model
        faces_found = test(os.path.join(CONFIG.TEST_PATH, secure_filename(image.filename)),
                           np.array(known_face_encodings), known_people)
        return jsonify(faces_found)

    else:
        return redirect(url_for('index'))


@app.route('/delete_face', methods=["GET", "POST"])
def delete_face():

    if request.method == 'POST':

        name = request.form['del_face_name']
        known_faces = {}
        with open(CONFIG.DB_JSON_PATH) as file:
            known_faces = json.load(file)

        # if the name entered was present in the database, then delete the data from the json file,
        # as well as delete the image from the directory
        if name in known_faces.keys():
            os.remove(known_faces[name]['image'])
            del known_faces[name]

            json.dump(known_faces, codecs.open(CONFIG.DB_JSON_PATH, 'w', encoding='utf-8'), separators=(',', ':'),
                      sort_keys=True, indent=4)

            # notify the user for successful transaction
            session['status'] = 'success'
            flash('The Face was successfully deleted from the database !')
            return redirect(url_for('index'))

        else:
            # notify the user for a failed transaction
            flash("Name was not found in the database. Enter the name correctly !")
            session['status'] = 'failed'
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/api/faces', methods=['POST', 'GET'])
def faces():
    if os.path.exists(CONFIG.DB_JSON_PATH):
        with open(CONFIG.DB_JSON_PATH) as file:
            data = json.load(file)
            return jsonify(list(data.keys()))

