import os

# Get the project root directory (parent of utils folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

def write_to_log(content, filename):

    os.makedirs(LOG_DIR, exist_ok=True)

    filepath = os.path.join(LOG_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Content written to {filepath}")
