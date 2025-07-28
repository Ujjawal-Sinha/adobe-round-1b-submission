from sentence_transformers import SentenceTransformer
import os

def download_model_to_project():
    """
    This script is run ONCE by you, locally, to download the model files
    directly into your project directory.
    """
    model_name = 'all-MiniLM-L6-v2'
    # We will save the model inside the 'models' directory
    save_path = os.path.join('models', model_name)

    print(f"Downloading model '{model_name}' to '{save_path}'...")

    os.makedirs(save_path, exist_ok=True)

    model = SentenceTransformer(model_name)
    model.save(save_path)

    print("\nModel downloaded and saved successfully!")
    print(f"The model files are now in the '{save_path}' directory.")
    print("You can now build your Docker image completely offline.")

if __name__ == "__main__":
    download_model_to_project()