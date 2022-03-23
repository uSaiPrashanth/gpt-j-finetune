from transformers import AutoModelForCausalLM,AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import Trainer,TrainingArguments
import torch
import lm_dataformat
import wandb
import torch.distributed as dist
import os
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.file_utils import cached_property,torch_required

CHECKPOINT = 1000

class LMDataset(torch.utils.data.IterableDataset):

    def __init__(self,path,max_length=2048,training=True,evaluation_steps=304):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        self.max_length = max_length
        self.training = training
        self.evaluation_steps = evaluation_steps

        self.iterator = iter(self.iterate())

    def __len__(self):
        """Returns length of P3 dataset

        > Length of train dataset is infinity
        > Length of test dataset is `evaluation_steps`
        """
        if(self.training):
            return 999999999999 # Batch size is irrelavant
        
        else:
            return self.evaluation_steps
    
    def iterate(self):
        """Iterator wrapper function

        Used in combination with `__iter__` continue iteration over subsequent text records on multiple calls
        """

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
            if(len(tokens) > (self.max_length)):
                yield self.parse(tokens[:self.max_length])

                tokens = tokens[self.max_length:]
                

    def parse(self,tokens):
        return {
            'input_ids':torch.tensor(tokens),
            'labels':torch.tensor(tokens)
        }

    def __iter__(self):
        if(self.training):
            yield from self.iterator
        
        else:
            for i in range(self.evaluation_steps):
                try:
                    yield next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.iterate())
                    yield next(self.iterator)
        
        
def attr(name):
    return getattr(self.module,name)

def model_init():
    model = AutoModelForCausalLM.from_pretrained(f'/mnt/ssd-1/P3_6B/checkpoint-{CHECKPOINT}')
    model.parallelize({
        0:list(range(2)),
        1:list(range(2,5)),
        2:list(range(5,9)),
        3:list(range(9,13)),
        4:list(range(13,17)),
        5:list(range(17,20)),
        6:list(range(20,24)),
        7:list(range(24,28))
    })
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    model = DDP(model)
    return model

class GPTJTrainingArguments(TrainingArguments):

    @property
    def place_model_on_device(self):
        return False

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        """Distributed Backend is initialized saperately. Do nothing"""

        device = torch.device("cuda:0")
        self._n_gpu = 1

        return device

class GPTJTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model.module.forward(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def _load_state_dict_in_model(self,state_dict):
        '''Model is loaded beforehand to avoid OOM. Do nothing'''
        return
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        
        if(checkpoint is None):
            return

        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint,"optimizer.pt"),map_location="cpu") # Loading on cuda causes OOM
        )
        
        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))


def train(args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.nnodes,
        rank=args.rank,
    )

    train_ds = LMDataset('/mnt/ssd-1/P3/P3_text/train')
    test_ds = LMDataset('/mnt/ssd-1/P3/P3_text/test',training=False)
    validation_ds = LMDataset('/mnt/ssd-1/P3/P3_text/validation',training=False)
    wandb.init(entity='eleutherai',project='gpt-j-finetune',group='P3_distributed')

    training_args = GPTJTrainingArguments(
        output_dir = './P3_6B/',
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        do_train=True,
        max_steps=5000,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=500,
        eval_steps=100,
        evaluation_strategy='steps',
        gradient_accumulation_steps = 16//args.nnodes,
        learning_rate=1.2e-5,
        lr_scheduler_type = 'cosine',
        local_rank=args.rank,
        report_to='wandb',
        run_name='pod-' + str(args.rank)
    )

    trainer = GPTJTrainer(
        model = model_init(),
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = test_ds,
    )
    trainer.train(f'/mnt/ssd-1/P3_6B/checkpoint-{CHECKPOINT}')
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--rank",type=int)
    parser.add_argument("--nnodes",type=int)
    args = parser.parse_args()
    
    train(args)