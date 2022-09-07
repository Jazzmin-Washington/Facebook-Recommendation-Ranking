#%% Clean and Resize Images
from PIL import ImageFile
from PIL import Image
from alive_progress import alive_bar
import os


class CleanImages():
    def __init__(self, final_image_size: int = 512):
        self.datapath = os.getcwd()
        self.final_image_size = final_image_size

    def create_save_folder(self):
        if not os.path.exists(f'{self.datapath}/data/clean_images'):
            os.makedirs (f'{self.datapath}/data/clean_images' , exist_ok=True)
            

    def clean_image(self, img_name : str):
            # open image
            img = Image.open(img_name)

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
        self.clean_image_save = f'{self.datapath}/data/clean_images'
        os.chdir(f'{self.datapath}/data/images')
        with alive_bar(len(os.listdir())) as bar:
            for image in os.listdir():
                if '.jpg' in image or '.jpeg' in image and image not in self.clean_image_save:
                    img  = self. clean_image(image)
                    img.save(f'{self.clean_image_save}/clean_{image}')
                    bar()
        return None

    def run_image_cleaner(self):
        self.create_save_folder()
        self.clean_all_images()

if __name__ == "__main__":
    cleaner = CleanImages()
    print("Let's begin reformatiing the images...")
    cleaner.run_image_cleaner()
    print(f"Success! Reformatted Images were saved at: {cleaner.clean_image_save} ")