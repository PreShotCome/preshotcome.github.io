import os
import subprocess

def download_model(file_id, destination):
    try:
        subprocess.check_call(['gdown', '--folder', '--id', file_id, '-O', destination])
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {destination}: {e}")

def main():
    os.makedirs('models', exist_ok=True)
    print("Downloading models...")

    # Positions Model
    download_model('1UxBFPzejBqBGk04QIy59O5Kf5yNYiGPb', 'models/positions.TensorFlow')

    # Watermark Model
    download_model('1hqsTAuTd71HbNUc58D2okVz_PldPaa3s', 'models/watermark.TensorFlow')

    # Genitals Model
    download_model('1_AjJbZY3gyxIdDkm3_kscHNp0J7u_bK-', 'models/genitals.TensorFlow')

    # Penetration Model
    download_model('13F20ClAKOS7SZ9PKrQXQiXlRYFRF97Ha', 'models/penetration.TensorFlow')

    print("All models downloaded successfully.")

if __name__ == '__main__':
    main()