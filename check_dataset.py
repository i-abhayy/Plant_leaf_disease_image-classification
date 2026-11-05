import os

dataset_path = r"C:\Users\ABHAY YADAV\OneDrive\Desktop\ghub\plan leaf disease system\Dataset\plantvillage dataset\color"

classes = [
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("Total categories:", len(classes))

for category in sorted(classes):
    print(category)