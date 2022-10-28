import torch, json
from PIL import Image
from inverse_vit.inverse_vit_model import InverseUSR
from inverse_vit.utils import *
from PIL import Image
from matplotlib import cm
from transformers import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def compute_metrics(eval_preds):
    weight = torch.tanh(torch.relu(model.image_pix)).squeeze().permute(1,2,0).detach().cpu().numpy()

    #nweight = (weight - wmin)/(wmax - wmin)

    im = Image.fromarray((weight*255).astype(np.uint8))
    im.save('data/gen_ims/last_im.png')

    image = (torch.tanh(torch.relu(model.image_pix)) - model.mean[:, None, None]) / model.std[:, None, None]

    out_sent = model.model.generate(image[None, :, :, :], **gen_kwargs)
    out_sent = model.tokenizer.decode(out_sent[0])
    print(out_sent)

    return {}


with open('config/conf.json', 'rt', encoding='utf-8') as ifile:
    config = json.load(ifile)
batch_size = config['batch_size']
device = config['device']
model = InverseUSR(device)

model.to(device)

prompt = config['start_prompt']

dataset = SingleContextDataset(
    model.tokenizer, prompt, 512, 1
)

collator = DummyCollator(model.tokenizer)

patience = 10
training_args = TrainingArguments(
    output_dir=f"models/inverse_vit/",
    overwrite_output_dir=True,
    do_eval=True,
    evaluation_strategy='steps',
    no_cuda=device == 'cpu',
    num_train_epochs=10000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=50000000,
    save_total_limit=patience,
    prediction_loss_only=False,
    warmup_steps=500,
    learning_rate=3e-1,
    ignore_data_skip=True,
    eval_steps=50,
    metric_for_best_model='eval_metric_score',
    greater_is_better=True,
    load_best_model_at_end=False,
    logging_steps=50
)

early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset,
    compute_metrics=compute_metrics,
    eval_dataset=dataset[:1],
    # callbacks=[early_stopping_cb]
)

trainer.train()

