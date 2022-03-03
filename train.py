from transformers import AutoModelForCausalLM,AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import Trainer,TrainingArguments
import torch
import lm_dataformat
import wandb
import os

class LMDataset(torch.utils.data.IterableDataset):

    def __init__(self,path,max_length=2048,training=True,evaluation_steps=304):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        self.max_length = max_length
        self.training = training
        self.evaluation_steps = evaluation_steps

    def parse(self,tokens):
        return {
            'input_ids':torch.tensor(tokens),
            'labels':torch.tensor(tokens)
        }

    def __iter__(self):
        workers = torch.utils.data.get_worker_info()
        if(workers):
            workers = workers.num_workers
        else:
            workers = 1
        
        reader = lm_dataformat.Reader(self.path)
        tokens = []
        step = 0
        for text in reader.stream_data():
            tokens.extend(
                self.tokenizer(
                    text,
                    return_tensors='np'
                ).input_ids[0].tolist()
            )
            tokens.append(self.tokenizer.eos_token_id)
            if(len(tokens) > (self.max_length)):
                yield self.parse(tokens[:self.max_length])

                tokens = tokens[self.max_length:]
                step+=1
                if(not self.training and step > self.evaluation_steps):
                    return
    

            

class GPTJTrainer(Trainer):
    def _load_state_dict_in_model(self,state_dict):
        '''Model is loaded beforehand to avoid OOM. Do nothing'''
        return
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint,"optimizer.pt"),map_location="cpu") # Loading on cuda causes OOM
        )
        
        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))




if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained('/mnt/ssd-1/P3_6B/checkpoint-1000')
    model.parallelize({
        0:list(range(2)),
        1:list(range(2,6)),
        2:list(range(6,9)),
        3:list(range(9,13)),
        4:list(range(13,17)),
        5:list(range(17,20)),
        6:list(range(20,24)),
        7:list(range(24,28))
    })
    
    
    train_ds = LMDataset('/mnt/ssd-1/P3/P3_text/train')
    test_ds = LMDataset('/mnt/ssd-1/P3/P3_text/test',training=False)
    validation_ds = LMDataset('/mnt/ssd-1/P3/P3_text/validation',training=False)
    wandb.init(entity='eleutherai',project='gpt-j-finetune',group='P3',reinit=True)

    training_args = TrainingArguments(
        output_dir = '/mnt/ssd-1/P3_6B/',
        overwrite_output_dir=True,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=8,
        do_train=True,
        max_steps=5000,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=500,
        eval_steps=100,
        evaluation_strategy='steps',
        gradient_accumulation_steps = 16,
        learning_rate=1.2e-5,
        lr_scheduler_type = 'cosine',
        report_to='wandb',
    )

    trainer = GPTJTrainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = test_ds,
    )
    trainer.train("/mnt/ssd-1/P3_6B/checkpoint-1000")
    trainer.evaluate()