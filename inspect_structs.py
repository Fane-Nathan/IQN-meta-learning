from tminterface.structs import SimStateData, CheckpointData
import inspect

print("Inspecting SimStateData...")
# Create a dummy object or inspect the class
print(dir(SimStateData))

# Check for specific keywords
keywords = ["collision", "contact", "hit", "wall", "touch"]
found = []
for attr in dir(SimStateData):
    if any(k in attr.lower() for k in keywords):
        found.append(attr)

if found:
    print(f"Found potential collision fields: {found}")
else:
    print("No obvious collision fields found.")
