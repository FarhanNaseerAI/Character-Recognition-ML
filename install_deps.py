import subprocess
import sys
import time

def install(package):
    print(f"Installing {package}...")
    for attempt in range(5):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir", "--default-timeout=1000"])
            print(f"Successfully installed {package}")
            return True
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt + 1} failed for {package}. Retrying in 5 seconds...")
            time.sleep(5)
    print(f"Failed to install {package} after 5 attempts.")
    return False

packages = [
    "tensorflow==2.17.0",
    "numpy==1.26.4",
    "Pillow==10.4.0",
    "scikit-learn==1.5.1",
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
    "jupyter==1.0.0",
    "Flask==3.0.3",
    "Werkzeug==3.0.3"
]

for pkg in packages:
    install(pkg)
