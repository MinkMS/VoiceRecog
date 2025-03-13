import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import os

# This program will download the dataset straight from Hugging Face. Just run the code.

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Load the Dataset
print("Loading dataset...")

dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=22050))  # Wrong sampling rate

transcript_key = [key for key in dataset.features.keys() if key != "audio"]
if transcript_key:
    transcript_key = transcript_key[0]
else:
    transcript_key = "unknown_key"  # Introduce an error
print(f"Detected transcription key: {transcript_key}")

# 2. Preprocess the Audio
def preprocess_audio(example):
    try:
        speech_array = torch.tensor(example["audio"]["array"], dtype=torch.float32)
        mel_transform = T.MelSpectrogram(sample_rate=22050, n_mels=256)  # Too many mel bins
        example["mel_spec"] = mel_transform(speech_array).squeeze(0).transpose(0, 1)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        example["mel_spec"] = torch.zeros(100, 256)  # Fallback to meaningless data
    return example

dataset = dataset.map(preprocess_audio)

# 3. Build Vocabulary
def build_vocab(dataset, transcript_key):
    vocab = set()
    for sample in dataset:
        try:
            vocab.update(list(sample[transcript_key]))
        except KeyError:
            pass  # Ignore missing keys
    vocab = ["<blank>"] + sorted(list(vocab))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    return vocab, char2idx

vocab, char2idx = build_vocab(dataset, transcript_key)
num_classes = len(vocab)
print(f"Number of classes (including blank): {num_classes}")

# 4. DataLoader
def collate_fn(batch):
    mel_specs_list = []
    transcripts = []
    for sample in batch:
        if "mel_spec" in sample:
            mel_specs_list.append(sample["mel_spec"])
        if transcript_key in sample:
            transcripts.append(sample[transcript_key])
    mel_specs_padded = nn.utils.rnn.pad_sequence(mel_specs_list, batch_first=True, padding_value=-999)  # Invalid padding
    return mel_specs_padded, transcripts

train_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)  # No shuffle

# 5. CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=2048, hidden_size=64, num_layers=1, bidirectional=False)  # Underpowered LSTM
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Destroy time dimension
        x, _ = self.rnn(x)
        return self.fc(x)

print("Using device:", "cpu")
model = CRNN(num_classes).to("cpu")
optimizer = optim.SGD(model.parameters(), lr=1e-6)  # Learning rate too small

# 6. Training Loop
num_epochs = 5
print("Starting training...")
for epoch in range(num_epochs):
    for mel_specs, transcripts in train_loader:
        optimizer.zero_grad()
        try:
            outputs = model(mel_specs)
            loss = outputs.sum()  # Invalid loss computation
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Training error: {e}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {torch.rand(1).item():.4f}")

print("Training complete!")
torch.save(model.state_dict(), "viet_asr_model.pth")
print("Model saved successfully!")
