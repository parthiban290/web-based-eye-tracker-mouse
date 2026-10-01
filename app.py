from flask import Flask, render_template, request, jsonify, session, Response, redirect, url_for
from flask_cors import CORS
import json
import os
import bcrypt
import pickle
from datetime import datetime, timedelta
from functools import wraps
import threading
import time
import numpy as np

# ==================== APP CONFIG ====================
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'eyeflow_secret_key_2026_parthiban'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=60)

DATA_DIR = 'data'
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
CALIBRATION_DIR = os.path.join(DATA_DIR, 'calibration')
SETTINGS_DIR = os.path.join(DATA_DIR, 'settings')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DIR, exist_ok=True)
os.makedirs(SETTINGS_DIR, exist_ok=True)

# Global tracker
tracker_instance = None
tracker_thread = None


# ==================== UTILITIES ====================
def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({"users": {}}, f)


def load_users():
    ensure_users_file()
    try:
        with open(USERS_FILE, 'r') as f:
            data = json.load(f)
            if 'users' not in data:
                data = {"users": {}}
            return data
    except Exception:
        data = {"users": {}}
        with open(USERS_FILE, 'w') as f:
            json.dump(data, f)
        return data


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def hash_password(password):
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password, password_hash):
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Not logged in'}), 401
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


def is_calibrated(username):
    cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')
    return os.path.exists(cal_file)


def get_default_settings():
    return {
        'cursor_smoothing': 0.18,
        'blink_threshold': 0.22,
        'sound_feedback': True,
        'voice_typing': True,
        'voice_language': 'en-US',
        'dwell_click': False,
        'scroll_gain': 600,
        'click_cooldown': 1.2,
        'scroll_deadzone': 0.035
    }


def load_user_settings(username):
    settings_file = os.path.join(SETTINGS_DIR, f'{username}.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except Exception:
            return get_default_settings()
    return get_default_settings()


def save_user_settings(username, settings):
    settings_file = os.path.join(SETTINGS_DIR, f'{username}.json')
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)


# ==================== AUTH ROUTES ====================
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        username = data.get('username', '').strip().lower()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        users = load_users()
        if username in users['users']:
            return jsonify({'error': 'Username already exists'}), 409

        users['users'][username] = {
            'username': username,
            'password_hash': hash_password(password),
            'created_date': datetime.now().isoformat(),
            'last_login': None
        }
        save_users(users)
        save_user_settings(username, get_default_settings())

        return jsonify({'message': 'Account created successfully'}), 201

    except Exception as e:
        print(f"Register error: {e}")
        return jsonify({'error': 'Server error during registration'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        username = data.get('username', '').strip().lower()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400

        users = load_users()
        if username not in users['users']:
            return jsonify({'error': 'Invalid username or password'}), 401

        user = users['users'][username]
        if not verify_password(password, user['password_hash']):
            return jsonify({'error': 'Invalid username or password'}), 401

        user['last_login'] = datetime.now().isoformat()
        save_users(users)

        session['username'] = username
        session.permanent = True

        calibrated = is_calibrated(username)

        return jsonify({
            'message': 'Login successful',
            'username': username,
            'calibrated': calibrated
        }), 200

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Server error during login'}), 500


@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    global tracker_instance, tracker_thread
    if tracker_instance:
        tracker_instance.stop()
        tracker_instance = None
        tracker_thread = None
    session.clear()
    return jsonify({'message': 'Logged out'}), 200


# ==================== SETTINGS ROUTES ====================
@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    settings = load_user_settings(session['username'])
    return jsonify(settings), 200


@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    try:
        data = request.get_json()
        username = session['username']
        current = load_user_settings(username)
        current.update(data)
        save_user_settings(username, current)

        global tracker_instance
        if tracker_instance and tracker_instance.running:
            tracker_instance.update_settings(current)

        return jsonify({'message': 'Settings saved'}), 200
    except Exception as e:
        print(f"Settings error: {e}")
        return jsonify({'error': 'Failed to save settings'}), 500


# ==================== CALIBRATION ROUTES ====================
@app.route('/api/calibration/status', methods=['GET'])
@login_required
def calibration_status():
    username = session['username']
    calibrated = is_calibrated(username)
    cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')

    result = {'calibrated': calibrated}
    if calibrated:
        try:
            with open(cal_file, 'rb') as f:
                cal_data = pickle.load(f)
            result['date'] = cal_data.get('calibrated_date', 'Unknown')
        except Exception:
            result['calibrated'] = False

    return jsonify(result), 200


@app.route('/api/calibration/save', methods=['POST'])
@login_required
def save_calibration():
    try:
        username = session['username']
        cal_data = request.get_json()

        if not cal_data:
            return jsonify({'error': 'No calibration data'}), 400

        required = ['eye_min_x', 'eye_min_y', 'eye_max_x', 'eye_max_y',
                     'nose_center_x', 'nose_center_y']
        for key in required:
            if key not in cal_data:
                return jsonify({'error': f'Missing field: {key}'}), 400

        save_data = {
            'username': username,
            'eye_min': [float(cal_data['eye_min_x']), float(cal_data['eye_min_y'])],
            'eye_max': [float(cal_data['eye_max_x']), float(cal_data['eye_max_y'])],
            'nose_center': [float(cal_data['nose_center_x']), float(cal_data['nose_center_y'])],
            'calibrated_date': datetime.now().isoformat(),
            'points_count': cal_data.get('points_count', 5)
        }

        cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')
        with open(cal_file, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Calibration saved for {username}: {save_data}")
        return jsonify({'message': 'Calibration saved successfully'}), 200

    except Exception as e:
        print(f"Calibration save error: {e}")
        return jsonify({'error': f'Failed to save calibration: {str(e)}'}), 500


@app.route('/api/calibration/delete', methods=['POST'])
@login_required
def delete_calibration():
    username = session['username']
    cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')
    if os.path.exists(cal_file):
        os.remove(cal_file)
    return jsonify({'message': 'Calibration deleted'}), 200


# ==================== TRACKER ROUTES ====================
@app.route('/api/tracker/start', methods=['POST'])
@login_required
def start_tracker():
    global tracker_instance, tracker_thread

    if tracker_instance and tracker_instance.running:
        return jsonify({'error': 'Tracker already running'}), 400

    username = session['username']

    if not is_calibrated(username):
        return jsonify({'error': 'Please calibrate first'}), 400

    settings = load_user_settings(username)

    cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')
    try:
        with open(cal_file, 'rb') as f:
            calibration = pickle.load(f)
    except Exception as e:
        return jsonify({'error': f'Failed to load calibration: {e}'}), 500

    try:
        from eye_tracker import EyeFlowTracker
        tracker_instance = EyeFlowTracker(settings=settings, calibration=calibration)
        tracker_thread = threading.Thread(target=tracker_instance.run, daemon=True)
        tracker_thread.start()

        time.sleep(1)

        if not tracker_instance.running:
            return jsonify({'error': 'Tracker failed to start. Check camera.'}), 500

        print(f"Tracker started for {username}")
        return jsonify({'message': 'Tracker started'}), 200

    except Exception as e:
        print(f"Tracker start error: {e}")
        return jsonify({'error': f'Failed to start tracker: {str(e)}'}), 500


@app.route('/api/tracker/stop', methods=['POST'])
@login_required
def stop_tracker():
    global tracker_instance
    if tracker_instance:
        tracker_instance.stop()
        time.sleep(0.5)
        tracker_instance = None
        return jsonify({'message': 'Tracker stopped'}), 200
    return jsonify({'error': 'No tracker running'}), 400


@app.route('/api/tracker/emergency', methods=['POST'])
@login_required
def emergency_stop():
    global tracker_instance, tracker_thread
    if tracker_instance:
        tracker_instance.emergency_stop()
        tracker_instance = None
        tracker_thread = None
    return jsonify({'message': 'Emergency stop executed'}), 200


@app.route('/api/tracker/status', methods=['GET'])
@login_required
def tracker_status():
    global tracker_instance
    if tracker_instance and tracker_instance.running:
        try:
            stats = tracker_instance.get_stats()
            return jsonify({
                'running': True,
                'fps': stats.get('fps', 0),
                'accuracy': stats.get('accuracy', 0),
                'latency': stats.get('latency', 0),
                'clicks': stats.get('clicks', 0),
                'scrolls': stats.get('scrolls', 0),
                'voice_inputs': stats.get('voice_inputs', 0),
                'uptime': stats.get('uptime', 0),
                'voice_active': stats.get('voice_active', False)
            }), 200
        except Exception as e:
            print(f"Status error: {e}")
    return jsonify({'running': False}), 200


@app.route('/api/tracker/voice/toggle', methods=['POST'])
@login_required
def toggle_voice():
    global tracker_instance
    if tracker_instance and tracker_instance.running:
        state = tracker_instance.toggle_voice()
        return jsonify({'voice_active': state}), 200
    return jsonify({'error': 'Tracker not running'}), 400


@app.route('/api/tracker/frame')
@login_required
def get_frame():
    global tracker_instance
    if tracker_instance and tracker_instance.running:
        frame_bytes = tracker_instance.get_frame_bytes()
        if frame_bytes:
            return Response(frame_bytes, mimetype='image/jpeg')
    return '', 204

# ==================== CALIBRATION CAPTURE ROUTES ====================

calibration_session = {}


def get_cal_session(username):
    global calibration_session
    if username not in calibration_session:
        calibration_session[username] = {
            'cap': None,
            'face_mesh': None,
            'points': [],
            'eye_min': [1.0, 1.0],
            'eye_max': [0.0, 0.0],
            'nose_center': [0.0, 0.0],
            'camera_ready': False
        }
    return calibration_session[username]


@app.route('/api/calibration/start-capture', methods=['POST'])
@login_required
def start_calibration_capture():
    import cv2 as cv
    import mediapipe as mp_module

    username = session['username']
    cal = get_cal_session(username)

    # Release old camera if exists
    if cal['cap'] is not None:
        try:
            cal['cap'].release()
        except:
            pass
        time.sleep(0.5)

    # Stop any running tracker that might hold the camera
    global tracker_instance
    if tracker_instance and tracker_instance.running:
        tracker_instance.stop()
        time.sleep(1)
        tracker_instance = None

    try:
        # Try multiple camera indices
        cap = None
        for cam_index in [0, 1, 2]:
            print(f"[Calibration] Trying camera index {cam_index}...")
            test_cap = cv.VideoCapture(cam_index)

            if test_cap.isOpened():
                # Set resolution
                test_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                test_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

                # Warm up camera - read and discard frames
                print(f"[Calibration] Camera {cam_index} opened, warming up...")
                for i in range(30):
                    ret, frame = test_cap.read()
                    if not ret:
                        time.sleep(0.05)
                    time.sleep(0.03)

                # Verify we can read a good frame
                ret, frame = test_cap.read()
                if ret and frame is not None and frame.size > 0:
                    cap = test_cap
                    print(f"[Calibration] Camera {cam_index} ready! Frame: {frame.shape}")
                    break
                else:
                    test_cap.release()
                    print(f"[Calibration] Camera {cam_index} no frames")
            else:
                print(f"[Calibration] Camera {cam_index} failed to open")

        if cap is None:
            return jsonify({'error': 'No camera found. Please check your webcam connection.'}), 500

        # Init MediaPipe
        face_mesh = mp_module.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Test face detection with warm frames
        face_found = False
        for attempt in range(20):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                face_found = True
                print(f"[Calibration] Face detected on attempt {attempt + 1}")
                break

            time.sleep(0.05)

        if not face_found:
            print("[Calibration] Warning: Face not found during warmup, but continuing...")

        # Store session
        cal['cap'] = cap
        cal['face_mesh'] = face_mesh
        cal['points'] = []
        cal['eye_min'] = [1.0, 1.0]
        cal['eye_max'] = [0.0, 0.0]
        cal['nose_center'] = [0.0, 0.0]
        cal['camera_ready'] = True

        print("[Calibration] Camera ready for calibration")
        return jsonify({
            'message': 'Camera ready',
            'face_detected': face_found
        }), 200

    except Exception as e:
        print(f"[Calibration] Start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Camera error: {str(e)}'}), 500


@app.route('/api/calibration/capture-point', methods=['POST'])
@login_required
def capture_calibration_point():
    import cv2 as cv

    username = session['username']
    cal = get_cal_session(username)

    cap = cal.get('cap')
    face_mesh = cal.get('face_mesh')

    if cap is None or face_mesh is None or not cal.get('camera_ready', False):
        return jsonify({
            'success': False,
            'error': 'Camera not ready. Click "Start Calibration" first.'
        }), 200

    if not cap.isOpened():
        return jsonify({
            'success': False,
            'error': 'Camera disconnected. Please restart calibration.'
        }), 200

    # Read and discard a few frames to get fresh data
    for _ in range(5):
        cap.read()
        time.sleep(0.02)

    # Now capture multiple samples
    iris_samples = []
    nose_samples = []
    max_attempts = 40
    attempts = 0

    while len(iris_samples) < 8 and attempts < max_attempts:
        ret, frame = cap.read()
        attempts += 1

        if not ret:
            time.sleep(0.05)
            continue

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            iris = lm[469]
            nose = lm[1]

            # Validate landmarks are reasonable
            if (0.1 < iris.x < 0.9 and 0.1 < iris.y < 0.9 and
                    0.1 < nose.x < 0.9 and 0.1 < nose.y < 0.9):
                iris_samples.append([iris.x, iris.y])
                nose_samples.append([nose.x, nose.y])

        time.sleep(0.03)

    print(f"[Calibration] Captured {len(iris_samples)} samples in {attempts} attempts")

    if len(iris_samples) < 3:
        return jsonify({
            'success': False,
            'error': f'Face detection failed ({len(iris_samples)}/8 samples). Please: 1) Face the camera directly, 2) Ensure good lighting, 3) Remove anything blocking your face, 4) Try again.'
        }), 200

    # Average the samples
    avg_iris = [float(np.mean([s[0] for s in iris_samples])),
                float(np.mean([s[1] for s in iris_samples]))]
    avg_nose = [float(np.mean([s[0] for s in nose_samples])),
                float(np.mean([s[1] for s in nose_samples]))]

    # Relative position (iris - nose)
    rel_x = avg_iris[0] - avg_nose[0]
    rel_y = avg_iris[1] - avg_nose[1]

    # Update min/max
    cal['eye_min'][0] = min(cal['eye_min'][0], rel_x)
    cal['eye_min'][1] = min(cal['eye_min'][1], rel_y)
    cal['eye_max'][0] = max(cal['eye_max'][0], rel_x)
    cal['eye_max'][1] = max(cal['eye_max'][1], rel_y)
    cal['nose_center'] = avg_nose

    point_data = {
        'iris': avg_iris,
        'nose': avg_nose,
        'rel': [rel_x, rel_y],
        'samples': len(iris_samples)
    }
    cal['points'].append(point_data)

    point_index = len(cal['points'])
    print(f"[Calibration] Point {point_index}/5 captured:")
    print(f"  iris=({avg_iris[0]:.4f}, {avg_iris[1]:.4f})")
    print(f"  nose=({avg_nose[0]:.4f}, {avg_nose[1]:.4f})")
    print(f"  rel=({rel_x:.4f}, {rel_y:.4f})")
    print(f"  eye_min=({cal['eye_min'][0]:.4f}, {cal['eye_min'][1]:.4f})")
    print(f"  eye_max=({cal['eye_max'][0]:.4f}, {cal['eye_max'][1]:.4f})")

    return jsonify({
        'success': True,
        'point_index': point_index,
        'samples_captured': len(iris_samples),
        'calibration_data': {
            'eye_min_x': cal['eye_min'][0],
            'eye_min_y': cal['eye_min'][1],
            'eye_max_x': cal['eye_max'][0],
            'eye_max_y': cal['eye_max'][1],
            'nose_center_x': cal['nose_center'][0],
            'nose_center_y': cal['nose_center'][1],
            'points_count': point_index
        }
    }), 200


@app.route('/api/calibration/finish', methods=['POST'])
@login_required
def finish_calibration():
    username = session['username']
    cal = get_cal_session(username)
    points = cal.get('points', [])

    if len(points) < 5:
        return jsonify({'error': f'Only {len(points)} points captured, need 5'}), 400

    # Add padding to eye range to prevent edge issues
    eye_min = cal['eye_min']
    eye_max = cal['eye_max']

    range_x = eye_max[0] - eye_min[0]
    range_y = eye_max[1] - eye_min[1]

    # Add 10% padding
    padding_x = range_x * 0.1
    padding_y = range_y * 0.1
    eye_min[0] -= padding_x
    eye_min[1] -= padding_y
    eye_max[0] += padding_x
    eye_max[1] += padding_y

    # Save calibration
    save_data = {
        'username': username,
        'eye_min': eye_min,
        'eye_max': eye_max,
        'nose_center': cal['nose_center'],
        'calibrated_date': datetime.now().isoformat(),
        'points_count': len(points),
        'points_raw': points
    }

    cal_file = os.path.join(CALIBRATION_DIR, f'{username}.pkl')
    with open(cal_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"[Calibration] === SAVED for {username} ===")
    print(f"  eye_min: {eye_min}")
    print(f"  eye_max: {eye_max}")
    print(f"  nose_center: {cal['nose_center']}")
    print(f"  range_x: {eye_max[0] - eye_min[0]:.6f}")
    print(f"  range_y: {eye_max[1] - eye_min[1]:.6f}")

    # Release camera
    if cal['cap'] is not None:
        try:
            cal['cap'].release()
            print("[Calibration] Camera released")
        except:
            pass

    # Clean up session
    calibration_session.pop(username, None)

    return jsonify({'message': 'Calibration saved successfully'}), 200


@app.route('/api/calibration/cancel', methods=['POST'])
@login_required
def cancel_calibration():
    username = session['username']
    cal = get_cal_session(username)

    if cal['cap'] is not None:
        try:
            cal['cap'].release()
        except:
            pass

    calibration_session.pop(username, None)
    return jsonify({'message': 'Calibration cancelled'}), 200

# ==================== PAGE ROUTES ====================
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/calibrate')
@login_required
def calibrate_page():
    return render_template('calibrate.html', username=session['username'])


@app.route('/dashboard')
@login_required
def dashboard():
    username = session['username']
    if not is_calibrated(username):
        return redirect(url_for('calibrate_page'))
    return render_template('dashboard.html', username=username)


@app.route('/tracker')
@login_required
def tracker():
    username = session['username']
    if not is_calibrated(username):
        return redirect(url_for('calibrate_page'))
    return render_template('tracker.html', username=username)


# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return redirect(url_for('index'))


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 50)
    print("  EyeFlow Server Starting...")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)