import cv2 as cv2
import numpy as np
from imageio.v2 import imread
from matplotlib import pyplot as plt
import imagehash 
from PIL import Image

def hash_car(path, new_width = 150, new_height = 100):
    try:
        # Load the image
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        # Convert to gray
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image
        resized_image = cv2.resize(gray_image, (new_width, new_height))

        plt.imshow(resized_image, cmap="gray")
        plt.show()

        # Convert to Python Image Library format
        pil_image = Image.fromarray(resized_image)

        # Hash the image
        hash_value = imagehash.dhash(pil_image)
        return hash_value

    except Exception as e:
        print(f"Error processing {path}: {e}")


# Dow sampling


image_paths = [
    'honda_civic_1.jpeg',
    'honda_civic_1.2.jpeg',
    'honda_civic_1.3.png',
    'honda_civic_2.jpeg',
    'honda_civic_close.jpeg'
]

hashd = {}
for path in image_paths:
    hashd[path] = hash_car(path)





# Print the dictionary containing hashes
print(hashd['honda_civic_1.jpeg']-hashd['honda_civic_1.2.jpeg'])
print(hashd['honda_civic_1.2.jpeg']-hashd['honda_civic_1.3.png'])
print(hashd['honda_civic_1.jpeg']-hashd['honda_civic_close.jpeg'])








# # Get the dimensions of the image
        # height, width = gray_image.shape

        # crop_height = int(gray_image.shape[0] * 0.8)
        # crop_width = int(gray_image.shape[1] * 0.8)

        # # Calculate the coordinates for cropping the center region
        # start_x = max(0, (width - crop_width) // 2)
        # start_y = max(0, (height - crop_height) // 2)
        # end_x = min(width, start_x + crop_width)
        # end_y = min(height, start_y + crop_height)

        # # Crop the image
        # cropped_image = image[start_y:end_y, start_x:end_x]
 
        # plt.imshow(resized_image, cmap='gray')
        # plt.show()








#plt.imshow(image1)


# # Example sizes of your images
# sizes = [(168, 300), (183, 275), (194, 259), (284, 558)]

# # Desired shape for all images
# desired_width = 200
# desired_height = 250

# for size in sizes:
#     # Read your images (replace with your image loading mechanism)
#     # image = cv2.imread('your_image_path.jpg')
    
#     # For this example, create a dummy image with the given size
#     dummy_image = cv2.resize(src=None, dsize=size)
    
#     # Resize the image to the desired shape
#     resized_image = cv2.resize(dummy_image, (desired_width, desired_height))





# # Load images


# #print("Image size:", image1.shape)

# # Calculate dHash for images
# hash1 = imagehash.dhash(image1)
# hash2 = imagehash.dhash(image2)
# hash3 = imagehash.dhash(image3)
# hash4 = imagehash.dhash(image4)
# hash5 = imagehash.dhash(image5)

# # Calculate the Hamming distance between the hashes
# hamming_distance = hash1 - hash5  # This gives the Hamming distance between the two hashes
# print(f"Hamming distance between the hashes: {hamming_distance}")



# Set all of the aspects, image ratio, and picture quality to be the same before hashing the image 