import time


print("Loading libraries...")
start = time.time_ns()

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset


print(f"Libraries loaded in {round((time.time_ns() - start) / 1000000, 3)} ms.")

print("Setting config...")
start = time.time_ns()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
lr = 10.0
batch_size = 128
step_size = 1
gamma = 0.9
# size of embedding in embedding bag
emsize = 2 ** 12 # 4096

print(f"Config set in {round((time.time_ns() - start) / 1000000, 3)} ms.")

print("Loading and preprocessing data...")
start = time.time_ns()

def preprocess_data(dataset, flip_labels=False):
    d1 = []
    for data in dataset:
        d2 = data["title"] + "\n" + data["text"]
        if flip_labels:
            d3 = int(not data["label"])
        else:
            d3 = data["label"]
        d1.append({"text": d2, "label": d3})
    return d1


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


gen = torch.Generator()
gen = gen.manual_seed(0)
dsg_test, dsg_train_ex = random_split(load_dataset("GonzaloA/fake_news")["test"], [0.5, 0.5], generator=gen)
dsm_test, dsm_train_ex = random_split(load_dataset("mohammadjavadpirhadi/fake-news-detection-dataset-english")["test"], [0.5, 0.5], generator=gen)
dsg_train = preprocess_data(load_dataset("GonzaloA/fake_news")["train"]) + preprocess_data(dsg_train_ex)
dsm_train = preprocess_data(load_dataset("mohammadjavadpirhadi/fake-news-detection-dataset-english")["train"], True) + preprocess_data(dsm_train_ex, True)


ds_test = NewsDataset(preprocess_data(dsg_test) + preprocess_data(dsm_test, True))
ds_train = NewsDataset(dsg_train + dsm_train)
ds_val = NewsDataset(preprocess_data(load_dataset("GonzaloA/fake_news")["validation"]))

print(f"Data loaded and preprocessed in {round((time.time_ns() - start) / 1000000, 3)} ms.")

print("Building vocabulary...")
start = time.time_ns()

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data):
    for entry in data:
        yield tokenizer(entry["text"])

vocab = build_vocab_from_iterator(yield_tokens(ds_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


print(f"Vocabulary built in {round((time.time_ns() - start) / 1000000, 3)} ms.")

print("Generating data batcher and iterator...")
start = time.time_ns()



def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for item in batch:
        _label = item["label"]
        _text = item["text"]
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

print(f"Data batcher and iterator generated in {round((time.time_ns() - start) / 1000000, 3)} ms.")

print("Creating model...")
start = time.time_ns()


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

model = TextClassificationModel(len(vocab), emsize, 2).to(device)

print(f"Model created in {round((time.time_ns() - start) / 1000000, 3)} ms.")

''' UNCOMMENT TO USE CUSTOM TRAINING PARAMETERS
custom_params = input("Do you want to use custom training parameters? (y/n): ")
if custom_params == "y":
    epochs = int(input("Enter number of epochs: "))
    lr = float(input("Enter learning rate: "))
    step_size = int(input("Enter step size: "))
    gamma = float(input("Enter gamma: "))
'''

print("Training model...")
print(f"Training params:", f"{epochs} epochs | LR = {lr} | Batch size = {batch_size} | Step size = {step_size} | Gamma = {gamma}")
print(f"Model params:", f"Vocabulary size = {len(vocab)} | Embedding size = {emsize} | Training device = {device}")
print("-" * 100)
start = time.time()

def train(dataloader):
    global optimizer, criterion, epoch
    model.train()
    total_acc, total_count = 0, 0
    log_interval = round(len(dataloader) / 2)
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"| Epoch: {epoch} | {idx}/{len(dataloader)} batches complete | Training time: {round(elapsed, 3)}s | Training accuracy: {round((total_acc / total_count) * 100, 3)}% | Current loss: {round(loss.item(), 3)} |")
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
total_accu = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(val_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 100)
    print(f"| End of epoch {epoch} | Elapsed time: {round(time.time() - epoch_start_time,3)}s | Validation accuracy: {round(accu_val * 100, 3)}% | Device temperature: {torch.cuda.temperature()}°C |")
    print("-" * 100)

print(f"Model trained in {round((time.time() - start), 3)} s.")

print("Testing model with test dataset...")
start = time.time_ns()

accu_test = evaluate(test_dataloader)
print(f"Test accuracy: {round(accu_test * 100, 3)}%")

print(f"Testing completed in {round((time.time_ns() - start) / 1000000, 3)} ms.")

try:
    with open("accuracy.txt", "r") as f:
        old_accu = float(f.read())
        if old_accu < accu_test:
            with open("accuracy.txt", "w") as f:
                f.write(str(accu_test))
                print(f"New model accuracy ({round(accu_test,5)*100}%) is better than old model accuracy ({round(old_accu,5)*100}%).")
                print("Saving model...")
                start = time.time_ns()

                torch.save(model.state_dict(), "model.pth")

                print(f"Model saved in {round((time.time_ns() - start) / 1000000, 3)} ms.")

        else:
            print(f"New model accuracy ({round(accu_test,5)*100}%) is not better than old model accuracy ({round(old_accu,5)*100}%).")
            print("Model not saved.")
except FileNotFoundError:
    with open("accuracy.txt", "w") as f:
        f.write(str(accu_test))
        print("No old model found. Saving model...")
        start = time.time_ns()

        torch.save(model.state_dict(), "model.pth")

        print(f"Model saved in {round((time.time_ns() - start) / 1000000, 3)} ms.")