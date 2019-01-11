import pickle
import PIL.Image
import torch
from torchvision import transforms
from .models import EncoderCNN, DecoderRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process(image, encoder_path, decoder_path, vocab, embedding_dim, hidden_dim, n_layers):
    # apply data transformation
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # load vocab
    with open(vocab, 'rb') as f:
        vocab = pickle.load(f)

    # build models
    encoder = EncoderCNN(embedding_dim)
    decoder = DecoderRNN(embedding_dim, hidden_dim, len(vocab), n_layers)
    # evaluation mode
    encoder.eval()
    decoder.eval()
    # move to cpu
    encoder.to(device)
    decoder.to(device)

    # load model
    checkpoint_encoder = torch.load(encoder_path)
    checkpoint_decoder = torch.load(decoder_path)
    encoder.load_state_dict(checkpoint_encoder)
    decoder.load_state_dict(checkpoint_decoder)

    # process image
    img = PIL.Image.open(image)
    img = data_transforms(img)
    img.unsqueeze(0)  # Add batch size for PyTorch: [B, C, H, W]

    # generate captions from the image
    extracted_features = encoder(img)
    out_idx = decoder.sample(extracted_features)
    out_idx = out_idx[0].numpy()

    # convert word ids to word
    sampled_caption = []
    for word_id in out_idx:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    output = ' '.join(sampled_caption)

    print(output)
    return output
