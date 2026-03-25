from osso.config import SamplingParams
from osso.engine.engine import Engine
from osso.engine.generate import generate

MODEL_PATH = "meta-llama/Llama-3.2-1B"

engine = Engine(MODEL_PATH)
print("Model loaded successfully.")

output = generate(engine, "The capital of France is", SamplingParams())
print(output)
