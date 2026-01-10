import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class OCRDataset:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = '/home/tanlytran/.cache/kagglehub/datasets/crawford/emnist/versions/3'
        
        self.training_data_path = os.path.join(base_path, 'emnist-balanced-train.csv')
        self.test_data_path = os.path.join(base_path, 'emnist-balanced-test.csv')
        
        # # Valid image extensions (for image)
        # self.valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        # # Load training and test data (for image)
        # self.train_data = self.load_data(self.training_data_path)
        # self.test_data = self.load_data(self.test_data_path)

        # for EMNIST CSV
        self.train_data = self.load_emnist_csv(self.training_data_path)
        self.test_data = self.load_emnist_csv(self.test_data_path)
        self.X_train, self.y_train = self.train_data
        self.X_test, self.y_test = self.test_data
        
    # def load_data(self, data_path): # (for image dataset)
    #     data = []
    #     if not os.path.exists(data_path):
    #         print(f"Warning: Path {data_path} does not exist")
    #         return data
            

    #     folder_list = os.listdir(data_path)
    #     for label in folder_list:
    #         images_path = os.path.join(data_path, label)
    #         if os.path.isdir(images_path):
    #             for image_name in os.listdir(images_path):

    #                 # Skip non-image files
    #                 file_ext = os.path.splitext(image_name)[1].lower()
    #                 if file_ext not in self.valid_extensions:
    #                     continue
                    
    #                 img_path = os.path.join(images_path, image_name)
    #                 # store both image and label path
    #                 data.append((img_path, label))
    #     return data

    def load_emnist_csv(self, csv_path, target_size=(28, 28)):    # for EMNIST 
        # Read CSV (First column = label, Rest = pixels)
        df = pd.read_csv(csv_path, header=None)
        
        # Extract
        y_data = df.iloc[:, 0].values
        X_data = df.iloc[:, 1:].values
        
        # Reshape: 784 pixels -> 28x28 image
        X_data = X_data.reshape(-1, 28, 28)
        
        # EMNIST CSV images are rotated 90 deg and flipped (match MatLab): we fix this here
        X_data = np.transpose(X_data, (0, 2, 1))
        
        # Add Channel Dimension: (N, 28, 28) -> (N, 28, 28, 1)
        X_data = X_data.reshape(-1, 28, 28, 1)
        
        # # RESIZE if needed
        # X_data = tf.image.resize(X_data, target_size).numpy()
        
        # Normalize: 0-255 -> 0-1
        X_data = X_data / 255.0
        
        return X_data, y_data
    
    def get_train_dataframe(self):
        return pd.DataFrame(self.train_data, columns=['image_path', 'label'])
    
    def get_test_dataframe(self):
        return pd.DataFrame(self.test_data, columns=['image_path', 'label'])
    
    def get_all_dataframe(self):
        train_df = self.get_train_dataframe()
        train_df['split'] = 'train'
        
        test_df = self.get_test_dataframe()
        test_df['split'] = 'test'
        
        return pd.concat([train_df, test_df], ignore_index=True)
    
    def get_data_augmentation(self):
        data_generation = ImageDataGenerator(
            rotation_range=15,          # rotation for better variation
            width_shift_range=0.1,     # Increased shift range
            height_shift_range=0.1,
            zoom_range=0.15,             # More zoom variation
            shear_range=0.2,           # Increased shear
            fill_mode='constant',       # prevents smearing artifacts
            cval=0                      # fills empty space with black (0)
        )
        return data_generation


    


