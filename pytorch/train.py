"""Contains code for training model"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from .models import EncoderCNN, DecoderRNN
from .dataset import get_loader


BATCH_SIZE = 32
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 1
EPOCHS = 10
LEARNING_RATE = 0.001
PRINT_FREQ = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# apply data transformation and augmentation
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# load vocab
with open('vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

dataloader = get_loader('data/images', 'data/captions', vocab, data_transforms, BATCH_SIZE, shuffle=True)

# build models
encoderCNN = EncoderCNN(EMBEDDING_DIM)
decoderRNN = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), N_LAYERS)
encoderCNN.to(device)
decoderRNN.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
# parameters of our architecture consists of params of encoder + decoder
params = list(decoderRNN.parameters()) + list(encoderCNN.linear.parameters()) + list(encoderCNN.bn.parameters())
optimizer = optim.Adam(params, LEARNING_RATE)


# training
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}/{EPOCHS}')

    # keep into account the loss at each iteration
    running_loss = 0.0

    for i, (images, captions, lengths) in enumerate(dataloader):
        # move both inputs and labels to gpu if available
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # forward pass
        extracted_features = encoderCNN(images)
        outputs = decoderRNN(extracted_features, captions, lengths)

        # compute loss
        loss = criterion(outputs, targets)

        # zero all the gradients of the tensors optimizer will update
        optimizer.zero_grad()

        # backward pass + update parameters
        loss.backward()
        optimizer.step()

        # statistics
        #         running_loss += loss.item() * images.size(0)
        if i % PRINT_FREQ == 0:
            print(f'Epoch: {epoch}/{EPOCHS}  Step: {i}/{len(dataloader)}  Loss: {loss.item()}')

            #     epoch_loss = running_loss / dataset_size
            #     print(f'Loss: {epoch_loss:.4f}')
