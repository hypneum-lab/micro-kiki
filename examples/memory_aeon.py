"""Example: Aeon Memory Palace write + recall."""
from src.memory.aeon import AeonPalace

aeon = AeonPalace(dim=3072)

e1 = aeon.write("ESP32-S3 I2C requires external pull-up resistors", domain="embedded")
e2 = aeon.write("I2C bus speed: 100kHz standard, 400kHz fast mode", domain="embedded", links=[e1])

results = aeon.recall("I2C pull-up resistors", top_k=2)
for ep in results:
    print(f"[{ep.domain}] {ep.content}")

print(f"Memory stats: {aeon.stats}")
