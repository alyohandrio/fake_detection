import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="images")
parser.add_argument("--train", type=str, default="train_images")
parser.add_argument("--test", type=str, default="test_images")
parser.add_argument("--validation", type=str, default="validation_images")
parser.add_argument("--test_size", type=float, default=0)
parser.add_argument("--val_size", type=float, default=0)
args = parser.parse_args()

real_path = os.path.join(args.data, "0")
fake_path = os.path.join(args.data, "1")
real_images = [name for name in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, name))]
fake_images = [name for name in os.listdir(fake_path) if os.path.isfile(os.path.join(fake_path, name))]

test_size = args.test_size
val_size = args.val_size
train_real, test_real = train_test_split(real_images, test_size=test_size)
train_real, val_real = train_test_split(train_real, test_size=val_size / (1 - test_size))
train_fake, test_fake = train_test_split(fake_images, test_size=test_size)
train_fake, val_fake = train_test_split(train_fake, test_size=val_size / (1 - test_size))

for directory in [args.train, args.test, args.validation]:
    for cl in "01":
        if not os.path.exists(os.path.join(directory, cl)):
            os.makedirs(os.path.join(directory, cl))

train_data = (train_real, train_fake)
val_data = (val_real, val_fake)
test_data = (test_real, test_fake)
for directory, images in zip([args.train, args.test, args.validation], [train_data, test_data, val_data]):
    target_real = os.path.join(directory, "0")
    target_fake = os.path.join(directory, "1")
    for image in images[0]:
        shutil.copy(os.path.join(real_path, image), os.path.join(target_real, image))
    for image in images[1]:
        shutil.copy(os.path.join(fake_path, image), os.path.join(target_fake, image))

