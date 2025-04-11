import os
from flask import Flask, render_template, send_from_directory, request, jsonify, url_for
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

app = Flask(__name__, static_folder="../static")  # Ensure this points to the correct static directory

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
    return render_template("dashboard.html")

@app.route("/fingerprint_projections")
def fingerprint_projections():
    # Load the fingerprint image dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/dataset')
    dataset_path = os.path.abspath(dataset_path)  # Convert to absolute path
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        return f"The dataset path '{dataset_path}' does not exist. Please add fingerprint images to the directory."

    # Sort image files by name in ascending order
    image_files = sorted(
        [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.bmp'))]
    )
    if not image_files:
        return f"No images found in the dataset path: {dataset_path}"

    dataset = load_images(image_files)
    processed_data = preprocess_images(dataset)

    # Initialize the projection model with the processed data
    model = ProjectionModel(data=processed_data)
    model.train_model()

    # Project fingerprint changes to ages 5 and 10 years
    projections_5_years = model.predict_age(processed_data, age=5)
    projections_10_years = model.predict_age(processed_data, age=10)

    # Extract fingerprint features (e.g., minutiae, ridge endings, ridge bifurcations)
    extracted_features = [
        {
            "image_name": os.path.basename(image_file),
            "image_path": f"/data/dataset/{os.path.basename(image_file)}",
            "number": os.path.basename(image_file).split("__")[0],
            "sex": os.path.basename(image_file).split("__")[1].split("_")[0],
            "hand": os.path.basename(image_file).split("__")[1].split("_")[1],
            "finger": os.path.basename(image_file).split("__")[1].split("_")[2],
            **extract_fingerprint_features(image)
        }
        for image_file, image in zip(image_files, processed_data)
    ]

    # Calculate changes in features for 5 and 10 years
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

    # Zip features and changes for 5 and 10 years
    zipped_changes = [
        (feature, change_5, change_10)
        for feature, (change_5, change_10) in zip(extracted_features, zip(changes_5_years, changes_10_years))
    ]

    # Directory to save changed fingerprint images
    changed_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/changed_images'))
    if not os.path.exists(changed_images_dir):
        os.makedirs(changed_images_dir)

    # Generate changed fingerprint images
    for feature, change_5, change_10 in zipped_changes:
        original_image_path = os.path.join(dataset_path, feature["image_name"])
        normalized_image_name = os.path.splitext(feature["image_name"])[0].lower()  # Remove extension and normalize
        changed_image_5_path = os.path.join(changed_images_dir, f"{normalized_image_name}_5_years.png")
        changed_image_10_path = os.path.join(changed_images_dir, f"{normalized_image_name}_10_years.png")

        try:
            # Open the original image
            original_image = Image.open(original_image_path)

            # Ensure the image is in a supported format (e.g., RGB)
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Create modified images for 5 and 10 years
            changed_image_5 = original_image.copy()
            draw_5 = ImageDraw.Draw(changed_image_5)
            draw_5.text((10, 10), "5 Years Projection", fill="red")  # Example modification
            changed_image_5.save(changed_image_5_path, format="PNG")  # Explicitly save as PNG

            changed_image_10 = original_image.copy()
            draw_10 = ImageDraw.Draw(changed_image_10)
            draw_10.text((10, 10), "10 Years Projection", fill="blue")  # Example modification
            changed_image_10.save(changed_image_10_path, format="PNG")  # Explicitly save as PNG
        except Exception as e:
            print(f"Error processing image {feature['image_name']}: {e}")

    # Apply pagination
    page = int(request.args.get("page", 1))
    page_size = 10
    start = (page - 1) * page_size
    end = start + page_size
    paginated_changes = zipped_changes[start:end]
    total_pages = (len(zipped_changes) + page_size - 1) // page_size

    # Include changed image paths in the paginated results
    paginated_changes_with_images = [
        {
            "feature": feature,
            "change_5": change_5,
            "change_10": change_10,
            "changed_image_5_path": f"/data/changed_images/{os.path.splitext(feature['image_name'])[0].lower()}_5_years.png",
            "changed_image_10_path": f"/data/changed_images/{os.path.splitext(feature['image_name'])[0].lower()}_10_years.png"
        }
        for feature, change_5, change_10 in paginated_changes
    ]

    # Render the projections and changes in an HTML template
    return render_template(
        "index.html",
        changes_with_images=paginated_changes_with_images,  # Ensure this matches the template
        total_pages=total_pages,
        current_page=page
    )

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
    # Database file path
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all data from the database
    cursor.execute('''
        SELECT image_name, image_path, minutiae, ridge_endings, ridge_bifurcations,
               changed_minutiae_5, changed_ridge_endings_5, changed_ridge_bifurcations_5,
               changed_minutiae_10, changed_ridge_endings_10, changed_ridge_bifurcations_10,
               changed_image_5_path, changed_image_10_path
        FROM fingerprint_data
    ''')
    rows = cursor.fetchall()
    conn.close()

    # Format data for the template
    fingerprints_data = [
        {
            "feature": {
                "image_name": row[0],
                "image_path": row[1],
                "minutiae": row[2],
                "ridge_endings": row[3],
                "ridge_bifurcations": row[4]
            },
            "change_5": {
                "minutiae": row[5],
                "ridge_endings": row[6],
                "ridge_bifurcations": row[7]
            },
            "change_10": {
                "minutiae": row[8],
                "ridge_endings": row[9],
                "ridge_bifurcations": row[10]
            },
            "changed_image_5_path": row[11],
            "changed_image_10_path": row[12]
        }
        for row in rows
    ]

    # Apply pagination
    page = int(request.args.get("page", 1))
    page_size = 10
    start = (page - 1) * page_size
    end = start + page_size
    paginated_data = fingerprints_data[start:end]
    total_pages = (len(fingerprints_data) + page_size - 1) // page_size

    # Render the fingerprints page
    return render_template(
        "fingerprints.html",
        changes_with_images=paginated_data,
        total_pages=total_pages,
        current_page=page
    )

@app.route("/refresh_database", methods=["GET", "POST"])
def refresh_database():
    if request.method == "GET":
        # Render a page with a button to trigger the refresh operation
        return render_template("refresh_database.html")

    # POST request: Perform the refresh operation
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/fingerprint_data.db'))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Delete all data from the table
        cursor.execute("DELETE FROM fingerprint_data")

        # Reset the sequence for the primary key
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='fingerprint_data'")

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

@app.route("/test_static")
def test_static():
    return url_for('static', filename='bootstrap.min.css')

if __name__ == "__main__":
    app.run(debug=True)