import os
import torch
from datetime import datetime
from transformers import (AutoConfig, AutoTokenizer, DataCollatorWithPadding,
						  TrainingArguments, BitsAndBytesConfig)
from transformers.models.esm import EsmForDownstream
from datasets import Dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import pandas as pd
import wandb
from datetime import datetime
import argparse
from functools import partial
import yaml
from metrics import ComputeRegMetrics, ComputeClsMetrics, ComputeMultiClsMetrics, ComputeTokenClsMetrics
from utils import print_trainable_parameters, CustomClsTrainer, CustomRegTrainer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='train config file path')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--resume', type=str, default=None)

args = parser.parse_args()

with open(args.config_path, 'r') as f:
	cfg = yaml.safe_load(f)

os.environ["WANDB_NOTEBOOK_NAME"] = 'train_search.py'
wandb_dir = os.path.abspath('results/{}'.format(cfg['project_name']))
os.makedirs(wandb_dir, exist_ok=True)
os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)


def prepare_model(cfg, accelerator):
	model_name = cfg['esm_config']['model_name']  # "facebook/esm2_t33_650M_UR50D"
	
	config = AutoConfig.from_pretrained(
		model_name,
		return_unused_kwargs=False,
		num_labels=1,
		_from_auto=True,
	)
	
	config.update({
		'training_setting': cfg['training_setting'],
		'head_config': {
			'in_features': cfg['esm_config']['in_features'],
			'hidden_features': cfg['esm_config']['hidden_features'],
			'output_features': cfg['dataset']['num_classes'],
			'attention_features': cfg['head_config']['attention_features'],
			'attention_heads': cfg['head_config']['attention_heads'],
			'dropout_rate': cfg['head_config']['dropout_rate'],
			'use_pooling': True if 'use_pooling' not in cfg['head_config'] else False},
		'output_attentions': cfg['esm_config']['output_attentions'],
		'contact_compression': cfg['esm_config']['contact_compression'],
	})
	
	if cfg['quantized']:
		quantization_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type='nf4',
			llm_int8_threshold=6.0,
			llm_int8_has_fp16_weight=False,
			llm_int8_skip_modules=['head', 'contact_head']
		)
		model = EsmForDownstream.from_pretrained(
			model_name,
			cfg['head_config']['head_name'],
			config.head_config,
			config=config,
			torch_dtype=torch.bfloat16,
			quantization_config=quantization_config,)#.to('cuda')
		
		model.head.multihead_attn_layer._reset_parameters()
		model.head.pooling._reset_parameters()

		model = prepare_model_for_kbit_training(model)

	else:
		model = EsmForDownstream.from_pretrained(
			model_name,
			cfg['head_config']['head_name'],
			config.head_config,
			config=config).to('cuda')
	
	if cfg['peft']:
		# Convert the model into a PeftModel
		peft_config = LoraConfig(
			task_type=TaskType.SEQ_CLS,
			inference_mode=False,
			r=32,
			lora_alpha=32,
			target_modules=["query", "key", "value", "dense"], # also maybe "dense_h_to_4h" and "dense_4h_to_h"
			lora_dropout=0.1,
			bias="all" # or "all" or "lora_only" 
		)
		model = get_peft_model(model, peft_config)
		print_trainable_parameters(model)

		for param in model.base_model.model.head.parameters():
			param.requires_grad = True
		if cfg['esm_config']['output_attentions']:
			for param in model.base_model.model.contact_head.parameters():
				param.requires_grad = True		
	else:
		for name, param in model.named_parameters():
			if 'head' not in name:
				param.requires_grad = False
	
	# Use the accelerator
	model = accelerator.prepare(model)

	print_trainable_parameters(model)

	return model, config

def prepare_dataset(cfg, accelerator):
	csv_root = cfg['dataset']['csv_root']
	train_csv = pd.read_csv(csv_root.format('train'))
	test_csv = pd.read_csv(csv_root.format('test'))
	train_sequences = train_csv.sequence.tolist()
	test_sequences = test_csv.sequence.tolist()
	# Tokenization
	tokenizer = AutoTokenizer.from_pretrained(cfg['esm_config']['model_name'])
	if 'classification' in cfg['goal']:
		if 'GO' in cfg['dataset']['dataname'] or 'EC' in cfg['dataset']['dataname']:
			train_labels = [list(map(int, t.split(' '))) for t in train_csv.tgt_cls]
			test_labels = [list(map(int, t.split(' '))) for t in test_csv.tgt_cls]
		elif cfg['dataset']['dataname'] == 'FoldClassification' or \
                        cfg['dataset']['dataname'] == 'Loc':
			train_labels = train_csv.tgt_cls.tolist()
			test_labels = test_csv.tgt_cls.tolist()
		elif cfg['dataset']['dataname'] == 'SSP':
			train_labels = [list(map(int, t.split(' '))) for t in train_csv.tgt_cls]
			test_labels = [list(map(int, t.split(' '))) for t in test_csv.tgt_cls]
			train_tokenized, test_tokenized = {'input_ids': [], 'attention_mask': []}, {'input_ids': [], 'attention_mask': []}
			for sample in train_sequences:
				tokenized = tokenizer.batch_encode_plus([sample], add_special_tokens=False, padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
				train_tokenized['input_ids'].extend(tokenized['input_ids'])
				train_tokenized['attention_mask'].extend(tokenized['attention_mask'])
			for sample in test_sequences:
				tokenized = tokenizer.batch_encode_plus([sample], add_special_tokens=False, padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
				test_tokenized['input_ids'].extend(tokenized['input_ids'])
				test_tokenized['attention_mask'].extend(tokenized['attention_mask'])
			train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
			test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
			
			train_dataset = accelerator.prepare(train_dataset)
			test_dataset = accelerator.prepare(test_dataset)

			return tokenizer, train_dataset, test_dataset
		
	elif cfg['goal'] == 'regression':
		train_labels = train_csv.tgt_reg.tolist()
		test_labels = test_csv.tgt_reg.tolist()
		if 'FLIP' in cfg['dataset']['dataname']:
			train_labels = (np.array(train_labels) / 100).tolist()
			test_labels = (np.array(test_labels) / 100).tolist()

	# max_sequence_length = tokenizer.model_max_length
	max_sequence_length = 800
	if max(train_csv.sequence.str.len().max(), test_csv.sequence.str.len().max()) < max_sequence_length:
		max_sequence_length = max(train_csv.sequence.str.len().max(), test_csv.sequence.str.len().max())
		print("MAX SEQUENCE LENGTH: ", max_sequence_length)

	train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
	test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
	
	train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
	test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
	
	train_dataset = accelerator.prepare(train_dataset)
	test_dataset = accelerator.prepare(test_dataset)

	return tokenizer, train_dataset, test_dataset

def train(cfg, wandb_cfg=None, lr=None):
	# Initialize accelerator and Weights & Biases
	accelerator = Accelerator()
	with wandb.init(config=wandb_cfg, project=cfg['project_name']):		
		ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		saveroot = os.path.join(wandb_dir, f"{ts}")
		os.makedirs(saveroot, exist_ok=True)

		model, config = prepare_model(cfg, accelerator)
		tokenizer, train_dataset, test_dataset = prepare_dataset(cfg, accelerator)
		if cfg['task'] == 'determined':
			# Training setup
			training_args = TrainingArguments(
				output_dir=saveroot,
				overwrite_output_dir=True,
				learning_rate=wandb.config.lr,
				lr_scheduler_type=wandb.config.lr_scheduler_type,
				gradient_accumulation_steps=config.training_setting["gradient_accumulation_steps"],
				per_device_train_batch_size=config.training_setting["per_device_train_batch_size"],
				per_device_eval_batch_size=config.training_setting["per_device_eval_batch_size"],
				num_train_epochs=wandb.config.num_train_epochs,
				eval_strategy="epoch", #"epoch",
				eval_on_start=False,
				save_strategy="epoch",
				eval_do_concat_batches=False,
				batch_eval_metrics=True,
				load_best_model_at_end=True,
				metric_for_best_model=cfg['metric_for_best_model'],
				greater_is_better=(cfg['metric_for_best_model'] != 'rmse'),
				push_to_hub=False,
				logging_dir=None,
				logging_first_step=False,
				logging_steps=200,
				save_total_limit=2,
				no_cuda=False,
				seed=cfg['seed'] if 'seed' in cfg else 42,
				fp16=cfg['fp16'] if not cfg['quantized'] else False,
				fp16_backend=cfg['fp16'] if not cfg['quantized'] else False,
				report_to='wandb',
				disable_tqdm=True,
				gradient_checkpointing=config.training_setting['gradient_checkpointing']
				)
		
		elif cfg['task'] == 'search' or cfg['task'] == 'grid_search':
			print("!!!!!! lr", lr)
			# Training setup
			training_args = TrainingArguments(
				output_dir=saveroot,
				overwrite_output_dir=True,
				learning_rate=lr, #wandb.config.lr,
				lr_scheduler_type='cosine',
				gradient_accumulation_steps=config.training_setting["gradient_accumulation_steps"],
				per_device_train_batch_size=config.training_setting["per_device_train_batch_size"],
				per_device_eval_batch_size=config.training_setting["per_device_eval_batch_size"],
				num_train_epochs=3,
				eval_strategy="epoch", #"epoch",
				eval_on_start=False,
				save_strategy="no", #"no",
				eval_do_concat_batches=False,
				batch_eval_metrics=True,
				load_best_model_at_end=False,
				metric_for_best_model=cfg['metric_for_best_model'],
				greater_is_better=(cfg['metric_for_best_model'] != 'rmse'),
				push_to_hub=False,
				logging_dir=None,
				logging_first_step=False,
				logging_steps=200,
				save_total_limit=2,
				no_cuda=False,
				seed=cfg['seed'] if 'seed' in cfg else 42,
				fp16=cfg['fp16'] if not cfg['quantized'] else False,
				fp16_backend=cfg['fp16'] if not cfg['quantized'] else False,
				report_to='wandb',
				disable_tqdm=True,
				gradient_checkpointing=config.training_setting['gradient_checkpointing']
				)

		# Initialize Trainer
		if cfg['goal'] == 'regression':
			trainer = CustomRegTrainer(
				aux_list=['head'],
				model=model,
				args=training_args,
				train_dataset=train_dataset,
				eval_dataset=test_dataset,
				tokenizer=tokenizer,
				data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
				compute_metrics=ComputeRegMetrics())
		elif 'classification' in cfg['goal']:
			if 'token' in cfg['goal']:
				trainer = CustomClsTrainer(
					aux_list=['head'],
					criterion=cfg['training_setting']['criterion'],
					model=model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=test_dataset,
					tokenizer=tokenizer,
					data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
					compute_metrics=ComputeTokenClsMetrics())
			else:
				trainer = CustomClsTrainer(
					aux_list=['head'],
					criterion=cfg['training_setting']['criterion'],
					model=model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=test_dataset,
					tokenizer=tokenizer,
					data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
					compute_metrics=ComputeMultiClsMetrics() if 'multi' in cfg['goal'] else ComputeClsMetrics())

		trainer.train(resume_from_checkpoint=args.resume)
		del model
		torch.cuda.empty_cache()

if cfg['task'] == 'determined':
	sweep_config = cfg['sweep_config']
	count = 1
	sweep_id = wandb.sweep(sweep_config, project=cfg['project_name'])
	wandb.agent(sweep_id, function=partial(train, cfg=cfg), count=count)
elif cfg['task'] == 'grid_search':
	count = 1
	train(cfg, lr=args.lr)
else:
	sweep_config = cfg['sweep_config']
	count = 9
	sweep_id = wandb.sweep(sweep_config, project=cfg['project_name'])
	wandb.agent(sweep_id, function=partial(train, cfg=cfg), count=count)

wandb.finish()
