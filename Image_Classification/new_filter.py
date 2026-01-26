from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os

def apply_pop_art_filter(image_path, output_path="pop_art_image.png"):
    try:
        img = Image.open(image_path)

        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Color(img_contrast)
        img_color = enhancer.enhance(1.5)

        edges = img_color.filter(ImageFilter.FIND_EDGES)
        img_pop = ImageOps.invert(edges)
        img_pop = Image.composite(img_color, img_pop, img_pop.convert('L'))

        plt.imshow(img_pop)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Pop Art Filter Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_pop_art{ext}"
        apply_pop_art_filter(image_path, output_file)