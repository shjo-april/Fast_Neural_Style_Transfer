import glob

from PIL import Image

class Unlabeled_Dataset:
    def __init__(self, root_dir, transform):
        self.image_paths = glob.glob(root_dir + '/*')
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i]).convert('RGB')
        image = self.transform(image)
        return image