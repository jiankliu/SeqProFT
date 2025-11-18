import os
import torch
from datetime import datetime
from transformers import DataCollatorWithPadding, TrainingArguments
from transformers.models.esm import EsmForDownstream
from transformers.models.esm.configuration_esm import EsmConfig
from datasets import Dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import wandb
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

os.environ["WANDB_NOTEBOOK_NAME"] = 'train_search_esmc.py'
wandb_dir = os.path.abspath('results/{}'.format(cfg['project_name']))
os.makedirs(wandb_dir, exist_ok=True)
os.environ['WANDB_DIR'] = wandb_dir

def prepare_model(cfg, accelerator):
	model_name = cfg['esm_config']['model_name']  # "facebook/esm2_t33_650M_UR50D"
	config = EsmConfig(
		hidden_size=cfg['esm_config']['in_features'],
		num_hidden_layers=cfg['esm_config']['num_hidden_layers'],
		num_attention_heads=cfg['esm_config']['num_attention_heads'],
		output_attentions=cfg['esm_config']['output_attentions'],
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
			'use_pooling': cfg['head_config'].get('use_pooling', True)
		},
		'contact_compression': cfg['esm_config']['contact_compression'],
		'initializer_range': 0.02,
		'peft': cfg['peft']
	})

	model = EsmForDownstream(
		config,
		head_type=cfg['head_config']['head_name'],
		head_config=config.head_config,
		model_name=model_name,
		model_type='esmc'
	).to('cuda')
	
	if cfg['peft']:
		# Convert the model into a PeftModel
		peft_config = LoraConfig(
			task_type=TaskType.SEQ_CLS,
			inference_mode=False,
			r=32,
			lora_alpha=32,
			target_modules=["qkv", "out_proj", "ffn.1", "ffn.3"],
			lora_dropout=0.1,
			bias="all"
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

def prepare_dataset(cfg, accelerator, model):
	csv_root = cfg['dataset']['csv_root']
	train_csv = pd.read_csv(csv_root.format('train'))
	test_csv = pd.read_csv(csv_root.format('test'))
	train_sequences = train_csv.sequence.tolist()
	test_sequences = test_csv.sequence.tolist()
	
	if 'classification' in cfg['goal']:
		if 'GO' in cfg['dataset']['dataname'] or 'EC' in cfg['dataset']['dataname']:
			train_labels = [list(map(int, t.split(' '))) for t in train_csv.tgt_cls]
			test_labels = [list(map(int, t.split(' '))) for t in test_csv.tgt_cls]
		elif cfg['dataset']['dataname'] in ['FoldClassification', 'Loc', 'EnzyACT']:
			train_labels = train_csv.tgt_cls.tolist()
			test_labels = test_csv.tgt_cls.tolist()
		elif cfg['dataset']['dataname'] == 'SSP':
			train_labels = [list(map(int, t.split(' '))) for t in train_csv.tgt_cls]
			test_labels = [list(map(int, t.split(' '))) for t in test_csv.tgt_cls]
			train_tokenized, test_tokenized = {'input_ids': [], 'attention_mask': []}, {'input_ids': [], 'attention_mask': []}
			for sample in train_sequences:
				tokenized = model.esm._original_esmc.cpu()._tokenize([sample])[0][1:-1]
				train_tokenized['input_ids'].append(tokenized)
				train_tokenized['attention_mask'].append([1] * len(tokenized))
			for sample in test_sequences:
				tokenized = model.esm._original_esmc.cpu()._tokenize([sample])[0][1:-1]
				test_tokenized['input_ids'].append(tokenized)
				test_tokenized['attention_mask'].append([1] * len(tokenized))
			train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
			test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
			
			train_dataset = accelerator.prepare(train_dataset)
			test_dataset = accelerator.prepare(test_dataset)

			return train_dataset, test_dataset
		
	elif cfg['goal'] == 'regression':
		train_labels = train_csv.tgt_reg.tolist()
		test_labels = test_csv.tgt_reg.tolist()
		if 'FLIP' in cfg['dataset']['dataname']:
			train_labels = (np.array(train_labels) / 100).tolist()
			test_labels = (np.array(test_labels) / 100).tolist()

	max_sequence_length = 800
	max_seq_len = max(train_csv.sequence.str.len().max(), test_csv.sequence.str.len().max())
	if max_seq_len < max_sequence_length:
		max_sequence_length = max_seq_len
		print("MAX SEQUENCE LENGTH: ", max_sequence_length)

	train_sequences = [s[:max_sequence_length - 2] for s in train_sequences]
	test_sequences = [s[:max_sequence_length - 2] for s in test_sequences]

	train_tokens = model.esm._original_esmc.cpu()._tokenize(train_sequences)
	test_tokens = model.esm._original_esmc.cpu()._tokenize(test_sequences)
	train_attn_mask = (train_tokens != 1).long()
	test_attn_mask = (test_tokens != 1).long()
	train_tokenized = {'input_ids': train_tokens, 'attention_mask': train_attn_mask}
	test_tokenized = {'input_ids': test_tokens, 'attention_mask': test_attn_mask}
	
	train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
	test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
	
	train_dataset = accelerator.prepare(train_dataset)
	test_dataset = accelerator.prepare(test_dataset)

	return train_dataset, test_dataset

def train(cfg, wandb_cfg=None, lr=None):
	# Initialize accelerator and Weights & Biases
	accelerator = Accelerator()
	with wandb.init(config=wandb_cfg, project=cfg['project_name']):
		ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		saveroot = os.path.join(wandb_dir, f"{ts}")
		os.makedirs(saveroot, exist_ok=True)

		model, config = prepare_model(cfg, accelerator)
		train_dataset, test_dataset = prepare_dataset(cfg, accelerator, model)
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
				seed=cfg.get('seed', 42),
				fp16=cfg['fp16'] if not cfg['quantized'] else False,
				fp16_backend=cfg['fp16'] if not cfg['quantized'] else False,
				report_to='wandb',
				disable_tqdm=True,
				gradient_checkpointing=config.training_setting['gradient_checkpointing']
			)
		
		elif cfg['task'] in ['search', 'grid_search']:
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
				seed=cfg.get('seed', 42),
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
				tokenizer=None,
				data_collator=None,
				compute_metrics=ComputeRegMetrics()
			)
		elif 'classification' in cfg['goal']:
			if 'token' in cfg['goal']:
				trainer = CustomClsTrainer(
					aux_list=['head'],
					criterion=cfg['training_setting']['criterion'],
					model=model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=test_dataset,
					tokenizer=None,
					data_collator=None,
					compute_metrics=ComputeTokenClsMetrics()
				)
			else:
				trainer = CustomClsTrainer(
					aux_list=['head'],
					criterion=cfg['training_setting']['criterion'],
					model=model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=test_dataset,
					tokenizer=None,
					data_collator=None,
					compute_metrics=ComputeMultiClsMetrics() if 'multi' in cfg['goal'] else ComputeClsMetrics()
				)

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