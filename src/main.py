import os
from flask import Flask, render_template, send_from_directory, request, jsonify, url_for, Response, session, stream_with_context  # Add this import for streaming responses and session management
from models.projection_model import ProjectionModel
from utils.helpers import load_images, preprocess_images, extract_fingerprint_features
from PIL import Image, ImageDraw  # Add this import for image processing
import sqlite3  # Add this import for SQLite database operations
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for server environments
import numpy as np  # Import numpy for statistical calculations
from threading import Thread  # Add this import for background processing

app = Flask(__name__, static_folder="../static")  # Ensure this points to the correct static directory
app.secret_key = 'supersecretkey'  # Add a secret key for session management

# Serve the dataset directory as a static folder
@app.route('/data/dataset/<path:filename>')
def dataset_static(filename):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))
    return send_from_directory(dataset_path, filename)

# Serve the changed images directory as a static folder
@app.route('/data/changed_images/<path:filename>')
def changed_images_static(filename):
    changed_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))
    return send_from_directory(changed_images_path, filename)

@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        fingerprint_projections_url=url_for('fingerprint_projections')  # Ensure this line is present
    )

@app.route("/fingerprint_projections", methods=["GET", "POST"])
def fingerprint_projections():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table to track processing information
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT UNIQUE,
            status TEXT,  -- "pending", "success", or "failed"
            error_message TEXT DEFAULT NULL
        )
    ''')

    # Populate the processing_status table with pending entries if not already populated
    cursor.execute("SELECT image_name FROM processing_status WHERE status = 'pending'")
    pending_images = cursor.fetchall()
    if not pending_images:
        cursor.execute("SELECT image_name FROM raw_fingerprints")
        raw_fingerprints = cursor.fetchall()
        for image_name, in raw_fingerprints:
            cursor.execute('''
                INSERT OR IGNORE INTO processing_status (image_name, status)
                VALUES (?, 'pending')
            ''', (image_name,))
        conn.commit()

    # Fetch the next pending image
    cursor.execute("SELECT image_name FROM processing_status WHERE status = 'pending' LIMIT 1")
    next_image = cursor.fetchone()
    if not next_image:
        conn.close()
        return jsonify({"status": "complete", "message": "All images have been processed."})

    image_name = next_image[0]

    # Fetch the image path from raw_fingerprints
    cursor.execute("SELECT image_path FROM raw_fingerprints WHERE image_name = ?", (image_name,))
    image_path = cursor.fetchone()
    if not image_path:
        cursor.execute('''
            UPDATE processing_status
            SET status = 'failed', error_message = 'Image path not found in raw_fingerprints'
            WHERE image_name = ?
        ''', (image_name,))
        conn.commit()
        conn.close()
        return jsonify({"status": "error", "message": f"Image path not found for {image_name}."})

    image_path = image_path[0]

    # Directory to save changed fingerprint images
    changed_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))
    if not os.path.exists(changed_images_dir):
        os.makedirs(changed_images_dir)

    try:
        # Load and preprocess the image
        image_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"..{image_path}"))
        image = Image.open(image_full_path).convert("RGB")
        processed_image = preprocess_images([np.array(image)])[0]

        # Initialize the projection model
        model = ProjectionModel(data=np.array([processed_image], dtype=np.float32))
        model.train_model()

        # Extract fingerprint features
        features = extract_fingerprint_features(processed_image)

        # Save changed images
        normalized_image_name = os.path.splitext(image_name)[0].lower()
        changed_image_5_path = os.path.join(changed_images_dir, f"{normalized_image_name}_5_years.png")
        changed_image_10_path = os.path.join(changed_images_dir, f"{normalized_image_name}_10_years.png")

        changed_image_5 = image.copy()
        draw_5 = ImageDraw.Draw(changed_image_5)
        draw_5.text((10, 10), "5 Years Projection", fill="red")
        changed_image_5.save(changed_image_5_path, format="PNG")

        changed_image_10 = image.copy()
        draw_10 = ImageDraw.Draw(changed_image_10)
        draw_10.text((10, 10), "10 Years Projection", fill="blue")
        changed_image_10.save(changed_image_10_path, format="PNG")

        # Update processing status
        cursor.execute('''
            UPDATE processing_status
            SET status = 'success', error_message = NULL
            WHERE image_name = ?
        ''', (image_name,))
        conn.commit()

        conn.close()
        return jsonify({"status": "success", "message": f"Processed {image_name} successfully."})

    except Exception as e:
        # Update processing status with failure
        cursor.execute('''
            UPDATE processing_status
            SET status = 'failed', error_message = ?
            WHERE image_name = ?
        ''', (str(e), image_name))
        conn.commit()
        conn.close()
        return jsonify({"status": "error", "message": f"Failed to process {image_name}: {e}"})

@app.route("/fingerprint_projections_stream")
def fingerprint_projections_stream():
    # Stream real-time logs for the progress page
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/dataset')
    dataset_path = os.path.abspath(dataset_path)
    image_files = sorted(
        [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.bmp'))]
    )
    if not image_files:
        yield "data: No images found in the dataset path.\n\n"
        return

    dataset = load_images(image_files)
    processed_data = preprocess_images(dataset)

    model = ProjectionModel(data=processed_data)
    model.train_model()

    extracted_features = [
        {
            "image_name": os.path.basename(image_file),
            "image_path": f"/data/dataset/{os.path.basename(image_file)}",
            **extract_fingerprint_features(image)
        }
        for image_file, image in zip(image_files, processed_data)
    ]

    changed_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))
    if not os.path.exists(changed_images_dir):
        os.makedirs(changed_images_dir)

    for feature in extracted_features:
        normalized_image_name = os.path.splitext(feature["image_name"])[0].lower()
        changed_image_5_path = os.path.join(changed_images_dir, f"{normalized_image_name}_5_years.png")
        changed_image_10_path = os.path.join(changed_images_dir, f"{normalized_image_name}_10_years.png")

        if os.path.exists(changed_image_5_path) and os.path.exists(changed_image_10_path):
            yield f"data: Skipping {feature['image_name']} (already processed)\n\n"
            continue

        try:
            original_image_path = os.path.join(dataset_path, feature["image_name"])
            original_image = Image.open(original_image_path)

            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            changed_image_5 = original_image.copy()
            draw_5 = ImageDraw.Draw(changed_image_5)
            draw_5.text((10, 10), "5 Years Projection", fill="red")
            changed_image_5.save(changed_image_5_path, format="PNG")

            changed_image_10 = original_image.copy()
            draw_10 = ImageDraw.Draw(changed_image_10)
            draw_10.text((10, 10), "10 Years Projection", fill="blue")
            changed_image_10.save(changed_image_10_path, format="PNG")

            yield f"data: Processed {feature['image_name']}\n\n"
        except Exception as e:
            yield f"data: Error processing {feature['image_name']}: {e}\n\n"

    yield "data: Processing complete\n\n"

# Route to add image files and features to the SQLite database
@app.route("/add_to_database", methods=["GET", "POST"])
def add_to_database():
    if request.method == "GET":
        # Render a simple page with a button to trigger the POST request
        return render_template("add_to_database.html")

    # Database file path
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))

    # Connect to the SQLite database (create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists and recreate it if necessary
    cursor.execute("PRAGMA table_info(fingerprint_data)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    required_columns = [
        "id", "image_name", "image_path", "minutiae", "ridge_endings", "ridge_bifurcations",
        "changed_minutiae_5", "changed_ridge_endings_5", "changed_ridge_bifurcations_5",
        "changed_minutiae_10", "changed_ridge_endings_10", "changed_ridge_bifurcations_10",
        "changed_image_5_path", "changed_image_10_path"
    ]

    if set(required_columns) != set(existing_columns):
        try:
            # Rename the old table
            cursor.execute("ALTER TABLE fingerprint_data RENAME TO fingerprint_data_old")

            # Create the new table with the updated schema
            cursor.execute('''
                CREATE TABLE fingerprint_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT,
                    image_path TEXT,
                    minutiae INTEGER,
                    ridge_endings INTEGER,
                    ridge_bifurcations INTEGER,
                    changed_minutiae_5 INTEGER,
                    changed_ridge_endings_5 INTEGER,
                    changed_ridge_bifurcations_5 INTEGER,
                    changed_minutiae_10 INTEGER,
                    changed_ridge_endings_10 INTEGER,
                    changed_ridge_bifurcations_10 INTEGER,
                    changed_image_5_path TEXT,
                    changed_image_10_path TEXT
                )
            ''')

            # Migrate data from the old table to the new table
            cursor.execute('''
                INSERT INTO fingerprint_data (
                    id, image_name, image_path, minutiae, ridge_endings, ridge_bifurcations
                )
                SELECT id, image_name, image_path, minutiae, ridge_endings, ridge_bifurcations
                FROM fingerprint_data_old
            ''')

            # Drop the old table
            cursor.execute("DROP TABLE fingerprint_data_old")
        except sqlite3.OperationalError as e:
            conn.rollback()
            return f"Error updating database schema: {e}"

    # Recompute the data
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))
    changed_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))

    # Ensure directories exist
    if not os.path.exists(dataset_path) or not os.path.exists(changed_images_dir):
        return "Dataset or changed images directory does not exist."

    image_files = sorted(
        [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.bmp'))]
    )
    if not image_files:
        return "No images found in the dataset path."

    dataset = load_images(image_files)
    processed_data = preprocess_images(dataset)

    # Extract features and calculate changes
    extracted_features = [
        {
            "image_name": os.path.basename(image_file),
            "image_path": f"/data/dataset/{os.path.basename(image_file)}",
            **extract_fingerprint_features(image)
        }
        for image_file, image in zip(image_files, processed_data)
    ]

    changes_5_years = [
        {
            "image_name": feature["image_name"],
            "minutiae": round(len(feature["minutiae"]) * (1 + 0.05 * i), 2),
            "ridge_endings": round(len(feature["ridge_endings"]) * (1 + 0.02 * i), 2),
            "ridge_bifurcations": round(len(feature["ridge_bifurcations"]) * (1 + 0.03 * i), 2)
        }
        for i, feature in enumerate(extracted_features)
    ]

    changes_10_years = [
        {
            "image_name": feature["image_name"],
            "minutiae": round(len(feature["minutiae"]) * (1 + 0.1 * i), 2),
            "ridge_endings": round(len(feature["ridge_endings"]) * (1 + 0.05 * i), 2),
            "ridge_bifurcations": round(len(feature["ridge_bifurcations"]) * (1 + 0.08 * i), 2)
        }
        for i, feature in enumerate(extracted_features)
    ]

    paginated_changes_with_images = [
        {
            "feature": feature,
            "change_5": change_5,
            "change_10": change_10,
            "changed_image_5_path": f"/data/changed_images/{os.path.splitext(feature['image_name'].lower())[0]}_5_years.png",
            "changed_image_10_path": f"/data/changed_images/{os.path.splitext(feature['image_name'].lower())[0]}_10_years.png"
        }
        for feature, (change_5, change_10) in zip(extracted_features, zip(changes_5_years, changes_10_years))
    ]

    # Insert data into the database
    for change in paginated_changes_with_images:
        feature = change["feature"]
        change_5 = change["change_5"]
        change_10 = change["change_10"]
        cursor.execute('''
            INSERT INTO fingerprint_data (
                image_name, image_path, minutiae, ridge_endings, ridge_bifurcations,
                changed_minutiae_5, changed_ridge_endings_5, changed_ridge_bifurcations_5,
                changed_minutiae_10, changed_ridge_endings_10, changed_ridge_bifurcations_10,
                changed_image_5_path, changed_image_10_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature["image_name"],
            feature["image_path"],
            len(feature["minutiae"]),
            len(feature["ridge_endings"]),
            len(feature["ridge_bifurcations"]),
            change_5["minutiae"],
            change_5["ridge_endings"],
            change_5["ridge_bifurcations"],
            change_10["minutiae"],
            change_10["ridge_endings"],
            change_10["ridge_bifurcations"],
            change["changed_image_5_path"],
            change["changed_image_10_path"]
        ))

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    return "Data successfully added to the database."

@app.route("/fingerprints")
def fingerprints():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all data from the database
    cursor.execute('''
        SELECT sample, gender, hand, finger, image_name, image_path, minutiae, ridge_endings, ridge_bifurcations,
               changed_minutiae_5, changed_ridge_endings_5, changed_ridge_bifurcations_5,
               changed_minutiae_10, changed_ridge_endings_10, changed_ridge_bifurcations_10,
               changed_image_5_path, changed_image_10_path
        FROM fingerprint_data
        ORDER BY sample ASC
    ''')
    rows = cursor.fetchall()
    conn.close()

    # Format data for the template
    fingerprints_data = [
        {
            "sample": row[0],
            "gender": row[1],
            "hand": row[2],
            "finger": row[3],
            "feature": {
                "image_name": row[4],
                "image_path": row[5],
                "minutiae": row[6],
                "ridge_endings": row[7],
                "ridge_bifurcations": row[8]
            },
            "change_5": {
                "minutiae": row[9],
                "ridge_endings": row[10],
                "ridge_bifurcations": row[11]
            },
            "change_10": {
                "minutiae": row[12],
                "ridge_endings": row[13],
                "ridge_bifurcations": row[14]
            },
            "changed_image_5_path": row[15],
            "changed_image_10_path": row[16]
        }
        for row in rows
    ]

    # Pagination logic
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 100))  # Default page size is 100
    total_records = len(fingerprints_data)
    total_pages = (total_records + page_size - 1) // page_size  # Calculate total pages
    page = max(1, min(page, total_pages))  # Ensure the current page is within bounds
    start = (page - 1) * page_size
    end = start + page_size
    paginated_data = fingerprints_data[start:end]

    # Render the fingerprints page
    return render_template(
        "fingerprints.html",
        changes_with_images=paginated_data,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size,
        total_records=total_records,
        max=max,  # Pass max function to the template
        min=min   # Pass min function to the template
    )

@app.route("/refresh_database", methods=["GET", "POST"])
def refresh_database():
    if request.method == "GET":
        # Render a page with a button to trigger the refresh operation
        return render_template("refresh_database.html")

    # Database file path
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Delete all data from the table
        # cursor.execute("DELETE FROM fingerprint_data")

        # Reset the sequence for the primary key
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='fingerprint_data'")

        # Recreate the fingerprint_data table with new fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprint_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample INTEGER,
                gender TEXT,
                hand TEXT,
                finger TEXT,
                image_name TEXT,
                image_path TEXT,
                minutiae INTEGER,
                ridge_endings INTEGER,
                ridge_bifurcations INTEGER,
                changed_minutiae_5 INTEGER,
                changed_ridge_endings_5 INTEGER,
                changed_ridge_bifurcations_5 INTEGER,
                changed_minutiae_10 INTEGER,
                changed_ridge_endings_10 INTEGER,
                changed_ridge_bifurcations_10 INTEGER,
                changed_image_5_path TEXT,
                changed_image_10_path TEXT
            )
        ''')

        # Recompute the data
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))
        changed_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))

        # Ensure directories exist
        if not os.path.exists(dataset_path) or not os.path.exists(changed_images_dir):
            return jsonify({"status": "error", "message": "Dataset or changed images directory does not exist."})

        image_files = sorted(
            [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.bmp'))]
        )
        if not image_files:
            return jsonify({"status": "error", "message": "No images found in the dataset path."})

        dataset = load_images(image_files)
        processed_data = preprocess_images(dataset)

        # Extract features and calculate realistic changes
        extracted_features = [
            {
                "image_name": os.path.basename(image_file),
                "image_path": f"/data/dataset/{os.path.basename(image_file)}",
                **extract_fingerprint_features(image)
            }
            for image_file, image in zip(image_files, processed_data)
        ]

        def parse_filename(filename):
            # Extract sample, gender, hand, and finger from the filename
            parts = filename.split("__")
            sample = int(parts[0]) if parts[0].isdigit() else None
            details = parts[1].split("_")
            gender = details[0] if len(details) > 0 else None
            hand = details[1] if len(details) > 1 else None
            finger = details[2] if len(details) > 2 else None
            return sample, gender, hand, finger

        def calculate_realistic_changes(base_value, years, growth_rate):
            # Use a logarithmic growth model for realistic changes
            return round(base_value * (1 + growth_rate * np.log1p(years)), 2)

        changes_5_years = [
            {
                "image_name": feature["image_name"],
                "minutiae": calculate_realistic_changes(len(feature["minutiae"]), 5, 0.02),
                "ridge_endings": calculate_realistic_changes(len(feature["ridge_endings"]), 5, 0.01),
                "ridge_bifurcations": calculate_realistic_changes(len(feature["ridge_bifurcations"]), 5, 0.015)
            }
            for feature in extracted_features
        ]

        changes_10_years = [
            {
                "image_name": feature["image_name"],
                "minutiae": calculate_realistic_changes(len(feature["minutiae"]), 10, 0.02),
                "ridge_endings": calculate_realistic_changes(len(feature["ridge_endings"]), 10, 0.01),
                "ridge_bifurcations": calculate_realistic_changes(len(feature["ridge_bifurcations"]), 10, 0.015)
            }
            for feature in extracted_features
        ]

        paginated_changes_with_images = [
            {
                "feature": feature,
                "change_5": change_5,
                "change_10": change_10,
                "changed_image_5_path": f"/data/changed_images/{os.path.splitext(feature['image_name'].lower())[0]}_5_years.png",
                "changed_image_10_path": f"/data/changed_images/{os.path.splitext(feature['image_name'].lower())[0]}_10_years.png"
            }
            for feature, (change_5, change_10) in zip(extracted_features, zip(changes_5_years, changes_10_years))
        ]

        # Insert data into the database
        for change in paginated_changes_with_images:
            feature = change["feature"]
            change_5 = change["change_5"]
            change_10 = change["change_10"]
            sample, gender, hand, finger = parse_filename(feature["image_name"])
            cursor.execute('''
                INSERT INTO fingerprint_data (
                    sample, gender, hand, finger, image_name, image_path, minutiae, ridge_endings, ridge_bifurcations,
                    changed_minutiae_5, changed_ridge_endings_5, changed_ridge_bifurcations_5,
                    changed_minutiae_10, changed_ridge_endings_10, changed_ridge_bifurcations_10,
                    changed_image_5_path, changed_image_10_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sample, gender, hand, finger, feature["image_name"], feature["image_path"],
                len(feature["minutiae"]), len(feature["ridge_endings"]), len(feature["ridge_bifurcations"]),
                change_5["minutiae"], change_5["ridge_endings"], change_5["ridge_bifurcations"],
                change_10["minutiae"], change_10["ridge_endings"], change_10["ridge_bifurcations"],
                change["changed_image_5_path"], change["changed_image_10_path"]
            ))

        # Commit changes
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": f"Error refreshing database: {e}"})
    finally:
        conn.close()

    return jsonify({"status": "success", "message": "Database successfully refreshed."})

@app.route("/statistics")
def statistics():
    # Database file path
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch data for statistical analysis
    cursor.execute('''
        SELECT image_name, minutiae, ridge_endings, ridge_bifurcations,
               changed_minutiae_5, changed_ridge_endings_5, changed_ridge_bifurcations_5,
               changed_minutiae_10, changed_ridge_endings_10, changed_ridge_bifurcations_10
        FROM fingerprint_data
    ''')
    rows = cursor.fetchall()
    conn.close()

    # Prepare data for the table
    table_data = [
        {
            "image_name": row[0].replace('.BMP', '').replace('.bmp', ''),  # Normalize filename
            "original": {
                "minutiae": round(row[1], 2),
                "ridge_endings": round(row[2], 2),
                "ridge_bifurcations": round(row[3], 2)
            },
            "after_5_years": {
                "minutiae": round(row[4], 2),
                "ridge_endings": round(row[5], 2),
                "ridge_bifurcations": round(row[6], 2)
            },
            "after_10_years": {
                "minutiae": round(row[7], 2),
                "ridge_endings": round(row[8], 2),
                "ridge_bifurcations": round(row[9], 2)
            },
            "difference_5_years": {
                "minutiae": round(row[4] - row[1], 2),
                "ridge_endings": round(row[5] - row[2], 2),
                "ridge_bifurcations": round(row[6] - row[3], 2)
            },
            "difference_10_years": {
                "minutiae": round(row[7] - row[1], 2),
                "ridge_endings": round(row[8] - row[2], 2),
                "ridge_bifurcations": round(row[9] - row[3], 2)
            },
            "difference_5_vs_10_years": {
                "minutiae": round(row[7] - row[4], 2),
                "ridge_endings": round(row[8] - row[5], 2),
                "ridge_bifurcations": round(row[9] - row[6], 2)
            }
        }
        for row in rows
    ]

    # Extract differences for statistical calculations
    differences_5_years = {
        "minutiae": [row["difference_5_years"]["minutiae"] for row in table_data],
        "ridge_endings": [row["difference_5_years"]["ridge_endings"] for row in table_data],  # Ensure key matches
        "ridge_bifurcations": [row["difference_5_years"]["ridge_bifurcations"] for row in table_data]
    }
    differences_10_years = {
        "minutiae": [row["difference_10_years"]["minutiae"] for row in table_data],
        "ridge_endings": [row["difference_10_years"]["ridge_endings"] for row in table_data],  # Ensure key matches
        "ridge_bifurcations": [row["difference_10_years"]["ridge_bifurcations"] for row in table_data]
    }
    differences_5_vs_10_years = {
        "minutiae": [row["difference_5_vs_10_years"]["minutiae"] for row in table_data],
        "ridge_endings": [row["difference_5_vs_10_years"]["ridge_endings"] for row in table_data],  # Ensure key matches
        "ridge_bifurcations": [row["difference_5_vs_10_years"]["ridge_bifurcations"] for row in table_data]
    }

    # Calculate summary statistics
    def calculate_statistics(data):
        return {
            "mean": round(np.mean(data), 2),
            "median": round(np.median(data), 2),
            "std_dev": round(np.std(data), 2)
        }

    summary_statistics = {
        "difference_5_years": {feature: calculate_statistics(values) for feature, values in differences_5_years.items()},
        "difference_10_years": {feature: calculate_statistics(values) for feature, values in differences_10_years.items()},
        "difference_5_vs_10_years": {feature: calculate_statistics(values) for feature, values in differences_5_vs_10_years.items()}
    }

    # Calculate False Rejection Rate (FRR)
    def calculate_frr(data, mean, std_dev):
        outliers = [value for value in data if value < mean - std_dev or value > mean + std_dev]
        return round(len(outliers) / len(data) * 100, 2)

    frr_statistics = {
        "difference_5_years": {
            feature: calculate_frr(differences_5_years[feature],
                                   summary_statistics["difference_5_years"][feature]["mean"],
                                   summary_statistics["difference_5_years"][feature]["std_dev"])
            for feature in differences_5_years
        },
        "difference_10_years": {
            feature: calculate_frr(differences_10_years[feature],
                                   summary_statistics["difference_10_years"][feature]["mean"],
                                   summary_statistics["difference_10_years"][feature]["std_dev"])
            for feature in differences_10_years
        }
    }

    # Calculate False Rejection Rate (FRR) for Original vs Changed
    frr_statistics["original_vs_changed"] = {
        feature: calculate_frr(
            differences_10_years[feature],  # Use 10 years as the final "changed" state
            summary_statistics["difference_10_years"][feature]["mean"],
            summary_statistics["difference_10_years"][feature]["std_dev"]
        )
        for feature in differences_10_years
    }

    # Calculate average FRR for all features
    frr_statistics["original_vs_changed"]["average"] = round(
        sum(frr_statistics["original_vs_changed"].values()) / len(frr_statistics["original_vs_changed"]), 2
    )

    # Generate Pie Chart for Feature Percentages
    feature_totals = {
        "Minutiae": sum(differences_10_years["minutiae"]),
        "Ridge Endings": sum(differences_10_years["ridge_endings"]),
        "Ridge Bifurcations": sum(differences_10_years["ridge_bifurcations"])
    }
    plt.figure(figsize=(6, 6))
    plt.pie(feature_totals.values(), labels=feature_totals.keys(), autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green'])
    plt.title("Feature Percentage (Original vs Changed)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Correctly format feature names for consistency
    features = ["minutiae", "ridge_endings", "ridge_bifurcations"]

    # Generate Column Chart for Summary Statistics
    means = [summary_statistics["difference_10_years"][feature]["mean"] for feature in features]
    std_devs = [summary_statistics["difference_10_years"][feature]["std_dev"] for feature in features]

    plt.figure(figsize=(8, 6))
    x = range(len(features))
    plt.bar(x, means, width=0.4, label='Mean', color='blue', align='center')
    plt.bar(x, std_devs, width=0.4, label='Std Dev', color='orange', align='edge')
    plt.xticks(x, ["Minutiae", "Ridge Endings", "Ridge Bifurcations"])  # Display human-readable labels
    plt.title("Summary Statistics (Original vs Changed)")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    column_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Render the statistics page
    return render_template(
        "statistics.html",
        table_data=table_data,
        summary_statistics=summary_statistics,
        frr_statistics=frr_statistics,
        pie_chart=pie_chart,
        column_chart=column_chart
    )

@app.route("/raw_fingerprints", methods=["GET", "POST"])
def raw_fingerprints():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the raw_fingerprints table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT UNIQUE,
            image_path TEXT
        )
    ''')

    # Populate the table with images from /data/dataset/ if it's empty
    cursor.execute("SELECT COUNT(*) FROM raw_fingerprints")
    if cursor.fetchone()[0] == 0:
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))
        if os.path.exists(dataset_path):
            image_files = [
                f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.bmp'))
            ]
            for image_file in image_files:
                image_path = f"/data/dataset/{image_file}"
                cursor.execute('''
                    INSERT OR IGNORE INTO raw_fingerprints (image_name, image_path)
                    VALUES (?, ?)
                ''', (image_file, image_path))
            conn.commit()

    if request.method == "POST":
        # Handle adding a new image
        image_name = request.form.get("image_name")
        image_path = request.form.get("image_path")
        if image_name and image_path:
            try:
                cursor.execute('''
                    INSERT INTO raw_fingerprints (image_name, image_path)
                    VALUES (?, ?)
                ''', (image_name, image_path))
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # Ignore duplicate entries

    # Pagination logic
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 100))
    offset = (page - 1) * page_size

    cursor.execute("SELECT COUNT(*) FROM raw_fingerprints")
    total_records = cursor.fetchone()[0]
    total_pages = (total_records + page_size - 1) // page_size

    cursor.execute("SELECT id, image_name, image_path FROM raw_fingerprints LIMIT ? OFFSET ?", (page_size, offset))
    raw_fingerprints = cursor.fetchall()
    conn.close()

    return render_template(
        "raw_fingerprints.html",
        raw_fingerprints=raw_fingerprints,
        current_page=page,
        total_pages=total_pages,
        page_size=page_size,
        max=max,  # Pass max function to the template
        min=min   # Pass min function to the template
    )

@app.route("/raw_fingerprints/delete/<int:id>", methods=["POST"])
def delete_raw_fingerprint(id):
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete the fingerprint by ID
    cursor.execute("DELETE FROM raw_fingerprints WHERE id = ?", (id,))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Fingerprint deleted successfully."})


@app.route("/raw_fingerprints/edit/<int:id>", methods=["POST"])
def edit_raw_fingerprint(id):
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Update the fingerprint details
    image_name = request.form.get("image_name")
    image_path = request.form.get("image_path")
    if image_name and image_path:
        cursor.execute('''
            UPDATE raw_fingerprints
            SET image_name = ?, image_path = ?
            WHERE id = ?
        ''', (image_name, image_path, id))
        conn.commit()

    conn.close()
    return jsonify({"status": "success", "message": "Fingerprint updated successfully."})

@app.route("/fingerprint_progress", methods=["GET"])
def fingerprint_progress():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch progress information
    cursor.execute("SELECT COUNT(*) FROM processing_status")
    total_images = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM processing_status WHERE status = 'success'")
    processed_images = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM processing_status WHERE status = 'failed'")
    failed_images = cursor.fetchone()[0]

    pending_images = total_images - processed_images - failed_images

    conn.close()

    return jsonify({
        "total": total_images,
        "processed": processed_images,
        "failed": failed_images,
        "pending": pending_images
    })

@app.route("/reprocess_failed", methods=["POST"])
def reprocess_failed():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Reset the status of failed images to 'pending'
    cursor.execute('''
        UPDATE processing_status
        SET status = 'pending', error_message = NULL
        WHERE status = 'failed'
    ''')
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Failed images have been reset to pending for reprocessing."})

if __name__ == "__main__":
    app.run(debug=True)