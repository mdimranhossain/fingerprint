# Fingerprint

This project aims to project fingerprint changes based on age using a sample dataset(https://www.kaggle.com/datasets/ruizgara/socofing). The application utilizes machine learning techniques to analyze fingerprint data and predict changes over a span of 5 and 10 years.

## Project Structure

```
fingerprint
├── src
│   ├── main.py               # Entry point of the application
│   ├── data
│   │   └── dataset # original fingerprint images from dataset
|   |   └── changed_images
|   |   └── fingerprint_data.db # SQLite database
│   ├── models
│   │   └── projection_model.py # Contains the ProjectionModel class
|   ├── templates  #html ui/templates
│   ├── utils
│   │   └── helpers.py         # Utility functions for data handling
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files to ignore in Git
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/mdimranhossain/fingerprint.git
   cd fingerprint
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will load the dataset, initialize the projection model, and project fingerprint changes for ages 5 and 10 years.

## Functionality

- **Data Loading**: The application reads fingerprint data from data directory using reference from database.
- **Model Training**: It trains a projection model using the loaded dataset.
- **Age Prediction**: The model predicts fingerprint changes for specified ages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.