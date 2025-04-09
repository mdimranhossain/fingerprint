import os
from flask import Flask, render_template, send_from_directory, request
from models.projection_model import ProjectionModel
from utils.helpers import load_images, preprocess_images, extract_fingerprint_features

app = Flask(__name__)

# Serve the dataset directory as a static folder
@app.route('/data/dataset/<path:filename>')
def dataset_static(filename):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset'))
    return send_from_directory(dataset_path, filename)

@app.route("/")
def index():
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

    # Apply pagination
    page = int(request.args.get("page", 1))
    page_size = 10
    start = (page - 1) * page_size
    end = start + page_size
    paginated_changes = zipped_changes[start:end]
    total_pages = (len(zipped_changes) + page_size - 1) // page_size

    # Render the projections and changes in an HTML template
    return render_template(
        "index.html",
        zipped_changes=paginated_changes,
        total_pages=total_pages,
        current_page=page
    )

if __name__ == "__main__":
    app.run(debug=True)