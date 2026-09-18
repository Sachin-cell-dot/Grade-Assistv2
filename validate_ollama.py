import argparse
from services.ollama_service import OllamaVisionService
from tools.scoring import deterministic_total

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to a real worksheet image")
args = parser.parse_args()
result = OllamaVisionService().extract_image(args.image)
print("REAL VISION VALIDATION SUCCEEDED")
print(f"Deterministic total: {deterministic_total(result)}")
print(result.model_dump_json(indent=2))
