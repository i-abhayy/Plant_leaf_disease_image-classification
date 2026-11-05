import os

dataset_path = r"C:\Users\ABHAY YADAV\OneDrive\Desktop\ghub\plan leaf disease system\Dataset\plantvillage dataset\color"

total_images = 0

for category in sorted(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):
        images = [
            f for f in os.listdir(category_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        count = len(images)
        total_images += count

        print(f"{category}: {count}")

print("\nTotal images:", total_images)