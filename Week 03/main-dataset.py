from dataset import MNISTDataset

train_dataset = MNISTDataset(
    images_path='data/mnist/train-images.idx3-ubyte',
    labels_path='data/mnist/train-labels.idx1-ubyte'
)

print(len(train_dataset))  
img, label = train_dataset[0]
print(img.shape, label)    
