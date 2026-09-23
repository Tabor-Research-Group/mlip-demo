"""Small, model-independent helpers for the workshop notebooks."""
from pathlib import Path
from datetime import datetime
import sys
import uuid

def start_exercise():
    """Create a fresh output directory."""
    data = Path(__file__).cwd()
    output = data / 'results' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Python: {sys.executable}\nInputs: {data}\nNew results: {output}')
    return data, output


