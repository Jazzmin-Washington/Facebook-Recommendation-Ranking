#%% Clean and Resize Images
from PIL import ImageFile
from PIL import Image
from alive_progress import alive_bar
import pandas as pd
import os


class CleanImages():
    def __init__(self, final_image_size: int = 156):
        os.chdir('/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/')
        self.datapath = os.getcwd()
        self.final_image_size = final_image_size
        
        

    def create_save_folder(self):
        if not os.path.exists(f'{self.datapath}/clean_images'):
            os.makedirs (f'{self.datapath}/clean_images' , exist_ok=True)
        

    def clean_image(self, img_name : str):
            # open image
            with Image.open(img_name) as img:

                # resize by finding the biggest side of the image and calculating ratio to resize by
                max_side_length = max(img.size)
                resize_ratio = self.final_image_size / max_side_length
                img_width = int(img.size[0]*resize_ratio)
                img_height = int(img.size[1]*resize_ratio)
                img = img.resize((img_width, img_height))

                # convert to rgb
                img = img.convert("RGB")

                # paste on black image
                final_img = Image.new(mode="RGB", size=(
                    self.final_image_size, self.final_image_size))
                final_img.paste(img, ((self.final_image_size - img_width) //
                                2, (self.final_image_size - img_height)//2))

                return final_img

    def clean_all_images(self):
            self.clean_image_save = f'{self.datapath}/clean_images'
            os.chdir(f'{self.datapath}/images')
            with alive_bar(len(os.listdir())) as bar:
                for image in os.listdir():
                    check = list(os.listdir(f'{self.clean_image_save}'))
                    if '.jpg' in image and image not in check:
                        img  = self. clean_image(image)
                        img.save(f'{self.clean_image_save}/clean_{image}')
                        bar()
            return None

    def run_image_cleaner(self):
        self.create_save_folder()
        self.clean_all_images()

if __name__ == "__main__":
    cleaner = CleanImages()
    print("Let's begin reformatting the images...")
    cleaner.run_image_cleaner()
    print(f"Success! Reformatted Images were saved at: {cleaner.clean_image_save} ")
