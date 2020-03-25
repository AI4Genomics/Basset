# The Dataloader is responsible to take the result of the Dataset class and prepare it for training by batching the inputs
#   or apply shuffling if we pass the train data or test data to it (ie. if train, shuffle argument should be True).
# It can also apply some sort of transformation to the data if we ask to
#   (eg, rotating the input images with slight angles in each training iteration so the network would see them as new images and won't overfit)

train_dataloader = DataLoader(train_dataset, batchsize, shuffle=False, num_workers=data_workers)