import os

# Get the project root directory (parent of utils folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

def write_to_log(content, filename, append=True):

    os.makedirs(LOG_DIR, exist_ok=True)

    filepath = os.path.join(LOG_DIR, filename)

    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(content)
        f.write('\n')  # Add newline for better readability

    action = "appended to" if append else "written to"
    print(f"Content {action} {filepath}")
