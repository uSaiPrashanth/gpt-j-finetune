from unittest.util import _MAX_LENGTH
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Trainer,TrainingArguments
import torch
import lm_dataformat
import os

class LMDataset(torch.utils.data.IterableDataset):

    def __init__(self,path,max_length=2048):
        self.path = path
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        self.max_length = max_length

    def parse(self,tokens):
        return {
            'input_ids':torch.tensor(tokens[:self.max_length]),
            'labels':torch.tensor(tokens[1:])
        }

    def __iter__(self):
        workers = torch.utils.data.get_worker_info()
        if(workers):
            workers = workers.num_workers
        else:
            workers = 1
        
        reader = lm_dataformat.Reader(self.path)
        tokens = []
        for text in reader.stream_data():
            tokens.extend(
                self.tokenizer(
                    text,
                    return_tensors='np'
                ).input_ids[0].tolist()
            )
            tokens.append(self.tokenizer.eos_token_id)
            if(len(tokens) > (self.max_length+1)):
                yield self.parse(tokens[:self.max_length+1])

                tokens = tokens[self.max_length:]
    

            





if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
    model.parallelize({
        0:list(range(3)),
        1:list(range(3,6)),
        2:list(range(6,9)),
        3:list(range(9,13)),
        4:list(range(13,17)),
        5:list(range(17,20)),
        6:list(range(20,24)),
        7:list(range(24,28))
    })
    os.environ['WANDB_DISABLED'] = "true"
    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    # text = "EleutherAI is: "
    # tokens = tokenizer(text,return_tensors='pt')
    # generated_output = model.generate(temprature=0.4)

    # gen_text = tokenizer.batch_decode(generated_output)[0]
    # print(gen_text)

    # dataloader = torch.utils.data.DataLoader(
    #     LMDataset('/mnt/ssd-1/P3/P3_text/train'),
    #     batch_size=16
    # )

    # for i in tqdm(iter(dataloader)):
    #     pass

    train_ds = LMDataset('/mnt/ssd-1/P3/P3_text/train')
    test_ds = LMDataset('/mnt/ssd-1/P3/P3_text/test')
    validation_ds = LMDataset('/mnt/ssd-1/P3/P3_text/validation')

    training_args = TrainingArguments(
        output_dir = './P3_6B',
        overwrite_output_dir=True,
        per_device_train_batch_size=3,
        do_train=True,
        max_steps=30000,
        num_train_epochs=1,

    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = test_ds,
    )

    trainer.train()