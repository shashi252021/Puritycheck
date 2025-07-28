import os
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
import cv2
import pytesseract
import wikipedia
import json
import sqlite3
import numpy as np
import logging

# --- Configuration ---
# Set the path to the Tesseract executable on your server
# IMPORTANT: CHANGE THIS PATH TO YOUR TESSERACT INSTALLATION ON THE SERVER
# Example for Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux/macOS (if installed via package manager): pytesseract.pytesseract.tesseract_cmd = 'tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Database for caching online ingredient data (will be created in the same directory as app.py)
CACHE_DB_NAME = 'ingredient_safety_cache.db'

app = Flask(__name__)
CORS(app)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection Management for Flask ---
def get_db():
    """
    Establishes a new database connection if one doesn't exist for the current request.
    Stores it in Flask's 'g' object. Also ensures the table exists.
    """
    if 'db' not in g:
        try:
            g.db = sqlite3.connect(
                CACHE_DB_NAME,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row # Allows accessing columns by name
            
            # Ensure table exists within this connection (per-request context)
            cursor = g.db.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingredients (
                    name TEXT PRIMARY KEY,
                    is_harmful INTEGER,
                    reason TEXT,
                    details_json TEXT
                )
            ''')
            g.db.commit()
            app.logger.info(f"Database connected and table ensured: {CACHE_DB_NAME}")
        except sqlite3.Error as e:
            app.logger.error(f"Database connection error: {e}")
            raise # Re-raise the exception to indicate a critical error
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """
    Closes the database connection at the end of the request.
    """
    db = g.pop('db', None)
    if db is not None:
        db.close()
        app.logger.info("Database connection closed.")

# --- Simulated Online Data Fetch (REPLACE WITH REAL API INTEGRATION) ---
def _fetch_harmful_ingredient_data_online(ingredient_name):
    """
    SIMULATES fetching ingredient safety from an online source.
    In a real app, this would be an actual API call (e.g., requests.get).
    For demonstration, we'll hardcode some harmful ingredients.
    """
    app.logger.info(f"Simulating API fetch for: {ingredient_name}")
    # Simulate network delay for realism
    # import time
    # time.sleep(0.5)

    ingredient_name_lower = ingredient_name.lower().strip()

    # SIMULATED DATA: Mark specific ingredients as harmful for demonstration
    simulated_harmful_list = {
        "parabens": "Potential endocrine disruptor.",
        "sulfates": "Can cause skin irritation and dryness.",
        "phthalates": "Associated with various health issues.",
        "formaldehyde": "Known carcinogen and allergen.",
        "fragrance": "Common allergen, often unlisted chemicals.",
        "bht": "Potential carcinogen.",
        "peg": "Often contaminated with harmful byproducts."
    }

    if ingredient_name_lower in simulated_harmful_list:
        return True, simulated_harmful_list[ingredient_name_lower], {"source": "Simulated DB", "risk_level": "High"}
    else:
        return False, "Generally considered safe.", {"source": "Simulated DB", "risk_level": "Low"}

def get_ingredient_safety(ingredient_name):
    """
    Fetches ingredient safety data, using cache or online source.
    """
    db = get_db() # Get the database connection for the current request
    cursor = db.cursor()
    ingredient_name_lower = ingredient_name.lower().strip()

    # 1. Check local cache
    try:
        cursor.execute("SELECT is_harmful, reason, details_json FROM ingredients WHERE name = ?", (ingredient_name_lower,))
        cached_data = cursor.fetchone()
        if cached_data:
            app.logger.info(f"Found '{ingredient_name_lower}' in cache.")
            return bool(cached_data['is_harmful']), cached_data['reason'], json.loads(cached_data['details_json']) if cached_data['details_json'] else {}
    except sqlite3.Error as e:
        app.logger.error(f"Error checking cache for '{ingredient_name_lower}': {e}")
        # Continue to fetch from online if cache read fails
        
    # 2. If not in cache or cache read failed, fetch from simulated online source
    is_harmful, reason, details = _fetch_harmful_ingredient_data_online(ingredient_name_lower)

    # 3. Store in cache
    try:
        cursor.execute("INSERT OR REPLACE INTO ingredients (name, is_harmful, reason, details_json) VALUES (?, ?, ?, ?)",
                       (ingredient_name_lower, int(is_harmful), reason, json.dumps(details)))
        db.commit()
        app.logger.info(f"Cached '{ingredient_name_lower}'.")
    except sqlite3.Error as e:
        app.logger.error(f"Error caching ingredient '{ingredient_name_lower}': {e}")

    return is_harmful, reason, details

def fetch_wikipedia_definition(ingredient):
    """Fetches a brief Wikipedia summary."""
    try:
        return wikipedia.summary(ingredient, sentences=2, auto_suggest=True, redirect=True)
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            return wikipedia.summary(e.options[0], sentences=2, auto_suggest=True, redirect=True)
        return None
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.WikipediaException):
        app.logger.warning(f"Wikipedia definition not found for '{ingredient}'.")
        return None
    except Exception as e:
        app.logger.error(f"An unexpected error occurred with Wikipedia for '{ingredient}': {e}")
        return None

def preprocess_image(image_data):
    """Applies image pre-processing steps for better OCR."""
    # Convert image data (bytes) to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        app.logger.error("Could not decode image data into OpenCV format.")
        return None

    # Increase DPI by scaling the image
    scale_factor = 3.0
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Denoising (Median blur is good for salt-and-pepper noise)
    denoised_image = cv2.medianBlur(gray_image, 3)

    # Thresholding (Otsu's method often works well for binarization)
    _, thresh_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh_image

@app.route('/')
def index():
    """Serves the main HTML page."""
    # This assumes index.html is in a 'templates' folder relative to app.py
    return render_template('index.html') 

@app.route('/scan', methods=['POST'])
def scan_image():
    if 'image' not in request.files:
        app.logger.warning("No image file provided in request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        app.logger.warning("No selected file in request.")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read image data from the uploaded file
            image_data = file.read()
            app.logger.info(f"Received image file: {file.filename}, size: {len(image_data)} bytes")

            # Basic Tesseract check
            try:
                pytesseract.get_tesseract_version()
                app.logger.info("Tesseract is callable.")
            except pytesseract.TesseractNotFoundError:
                app.logger.critical("Tesseract OCR engine not found or path is incorrect.")
                return jsonify({"error": "Tesseract OCR engine not found on server. Please ensure it's installed and configured."}), 500

            # Pre-process image
            processed_image = preprocess_image(image_data)
            if processed_image is None:
                app.logger.error("Image pre-processing returned None.")
                return jsonify({"error": "Could not process image. Invalid format or corrupted."}), 400

            # Perform OCR
            custom_config = r'--oem 3 --psm 6'
            app.logger.info("Performing OCR...")
            product_text = pytesseract.image_to_string(processed_image, config=custom_config)
            app.logger.info(f"OCR extracted text length: {len(product_text.strip())}")

            if not product_text.strip():
                app.logger.info("No text detected in the image.")
                return jsonify({
                    "result_status": "no_text_detected",
                    "message": "No text detected in the image. Try a clearer image.",
                    "raw_text": ""
                }), 200

            # Tokenize and check safety
            potential_ingredients = set()
            delimiters = [',', '.', ';', '(', ')', '[', ']', '/', '\\', '\n', '\r', ':', '+', '*', '-', '=']
            temp_text = product_text
            for delim in delimiters:
                temp_text = temp_text.replace(delim, ' ')

            words = temp_text.split()
            for word in words:
                cleaned_word = ''.join(filter(str.isalpha, word)).strip().lower()
                if len(cleaned_word) > 2 and cleaned_word not in ['and', 'or', 'the', 'of', 'for', 'with', 'extract', 'powder', 'acid', 'salt', 'oil']:
                    potential_ingredients.add(cleaned_word)
            
            app.logger.info(f"Identified {len(potential_ingredients)} potential ingredients.")

            unsafe_ingredients_found = [] # List of dicts for JSON output

            sorted_potential_ingredients = sorted(list(potential_ingredients))

            for ingredient in sorted_potential_ingredients:
                is_harmful, reason, details = get_ingredient_safety(ingredient)
                if is_harmful:
                    wiki_def = fetch_wikipedia_definition(ingredient)
                    unsafe_ingredients_found.append({
                        "name": ingredient.capitalize(),
                        "reason": reason,
                        "wikipedia_definition": wiki_def if wiki_def else "Definition not found.",
                        "details": details
                    })
            
            app.logger.info(f"Found {len(unsafe_ingredients_found)} unsafe ingredients.")

            if unsafe_ingredients_found:
                response_message = "The Product is Not Recommended due to UNSAFE Ingredients."
                status = "unsafe"
            else:
                response_message = "The Product is SAFE to Use. All Identified Ingredients are Safe."
                status = "safe"

            return jsonify({
                "result_status": status,
                "message": response_message,
                "raw_text": product_text.strip(), # Still send raw_text from backend, even if not displayed
                "unsafe_ingredients": unsafe_ingredients_found
            }), 200

        except Exception as e:
            app.logger.error(f"An unexpected error occurred during image processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    # Ensure the 'templates' folder exists for render_template
    if not os.path.exists('templates'):
        os.makedirs('templates')
        app.logger.info("Created 'templates' directory.")

    # Run the Flask app
    # debug=True enables reloader and debugger, useful for development
    # host='0.0.0.0' makes the server accessible from other machines on the network
    app.run(debug=True, host='0.0.0.0', port=5000)
    # The database connection is now managed per-request by get_db and close_db
    # No need for a global CONN.close() here as it's handled by teardown_appcontext.
