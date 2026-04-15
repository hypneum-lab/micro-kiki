"""Example: RBD bias detection."""
import asyncio
import json
from src.cognitive.rbd import ReasoningBiasDetector

async def main():
    async def mock_detector(prompt):
        return json.dumps({"biased": False, "bias_type": None, "explanation": "Clean", "confidence": 0.05})

    rbd = ReasoningBiasDetector(generate_fn=mock_detector)
    result = await rbd.detect("How does I2C work?", "I2C uses SDA and SCL lines...")
    print(f"Biased: {result.biased}, Confidence: {result.confidence}")

asyncio.run(main())
