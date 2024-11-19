import cv2
import os
import albumentations as A
from glob import glob
from tqdm import tqdm


original_data_dir = r'C:\Users\Crain\Desktop\dataset'
augmented_data_dir = r'C:\Users\Crain\Desktop\augmented_dataset'


os.makedirs(augmented_data_dir, exist_ok=True)


augmentations = A.Compose([
    A.GaussNoise(p=0.2),               
    A.MotionBlur(blur_limit=3, p=0.2),  
    A.GaussianBlur(blur_limit=3, p=0.2), 
    A.RandomFog(p=0.2),                 
    A.RandomSnow(p=0.2),                
    A.ImageCompression(quality_lower=30, quality_upper=50, p=0.2),  
    A.PixelDropout(dropout_prob=0.2, p=0.2)  
])


def augment_and_save_images():
    for class_name in os.listdir(original_data_dir):
        class_dir = os.path.join(original_data_dir, class_name)
        output_class_dir = os.path.join(augmented_data_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        for img_path in tqdm(glob(os.path.join(class_dir, '*.jpg'))): 
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            
            
            augmented = augmentations(image=image)
            augmented_image = augmented['image']
            
            
            img_name = os.path.basename(img_path)
            augmented_img_path = os.path.join(output_class_dir, 'aug_' + img_name)
            cv2.imwrite(augmented_img_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))  #

augment_and_save_images()
print("Augmentation complete. Augmented images saved in:", augmented_data_dir)
