import torch
import os

class TSNDataSet(data.Dataset):
	def __init__(self, transform):
		self.transform = transform

	def __getitem__(self, index):
		record = self.video_list[index]
		# get file_name


		# make sure file exists

		# get indices

		# execute get command


	def get(filename):

		# get images
		

		# get audio_sequence


		pass

		images = list()
        
        print(indices)
        for seg_ind in indices:
            p = int(seg_ind)

            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # apply transform to images
        process_data = self.transform(images)
        return process_data, record.label


	def __len__(self):
		pass


def create_dataloader():
    root_path = '/home/mbc2004/datasets/Something-Something/frames/'
    train_list = '/home/mbc2004/datasets/Something-Something/annotations/val_videofolder.txt'
        
    num_segments = 8
    batch_size = 8
    workers = 16

	return torch.utils.data.DataLoader(
            TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl='{:06d}.jpg',
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])]),
                       Stack(roll=(False)),
                       ToTorchFormatTensor(div=(True)),
                       IdentityTransform(),
                   ]), dense_sample=False),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            drop_last=True)

