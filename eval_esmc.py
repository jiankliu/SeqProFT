import os
from datetime import datetime
from transformers import (AutoConfig, AutoTokenizer, DataCollatorWithPadding,
						  TrainingArguments)
from transformers.models.esm import EsmForDownstream
from transformers.models.esm.configuration_esm import EsmConfig

from datasets import Dataset
from peft import PeftModel
import pandas as pd
from datetime import datetime
import argparse
import yaml
from metrics import (ComputeRegMetrics,
        ComputeClsMetrics, ComputeMultiClsMetrics, ComputeTokenClsMetrics,
        ComputeClsMetricsEval, ComputeMultiClsMetricsEval, ComputeTokenClsMetricsEval)
from utils import CustomClsTrainer, CustomRegTrainer
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='train config file path')
parser.add_argument('ckpt_path', help='path to trained model')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
	cfg = yaml.safe_load(f)

def prepare_model(cfg, ckpt_path):
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
		'peft': cfg['peft'],
		'goal': cfg['goal']
	})
	
	model = EsmForDownstream(
		config,
		head_type=cfg['head_config']['head_name'],
		head_config=config.head_config,
		model_name=model_name,
		model_type='esmc'
	).to('cuda')

	if cfg['peft']:
		model = PeftModel.from_pretrained(model, ckpt_path)

	sd = torch.load(os.path.join(ckpt_path, 'aux_weights.pt'), map_location=model.device)
	model.load_state_dict(sd, strict=False)
	return model, config

def prepare_dataset(cfg, model):
	csv_root = cfg['dataset']['csv_root']
	train_csv = pd.read_csv(csv_root.format('train'))
	test_csv = pd.read_csv(csv_root.format('test'))
	if cfg['dataset']['dataname'] == 'FoldClassification' \
			or cfg['dataset']['dataname'] == 'SSP' \
			or cfg['dataset']['dataname'] == 'TAPE-flu':
		test_csv = pd.read_csv(csv_root.format(cfg['dataset']['test_set']))
	train_sequences = train_csv.sequence.tolist()
	test_sequences = test_csv.sequence.tolist()
	
	# Tokenization
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

			return train_dataset, test_dataset
		
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
	
	return train_dataset, test_dataset

def eval(cfg):
	ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	saveroot = os.path.join('results', cfg['project_name'], 'eval', f"{ts}")
	os.makedirs(saveroot, exist_ok=True)

	model, config = prepare_model(cfg, args.ckpt_path)
	train_dataset, test_dataset = prepare_dataset(cfg, model)

	# Training setup
	training_args = TrainingArguments(
		output_dir=saveroot,
		overwrite_output_dir=True,
		learning_rate=cfg['sweep_config']['parameters']['lr']['value'],
		per_device_train_batch_size=config.training_setting["per_device_train_batch_size"],
		per_device_eval_batch_size=config.training_setting["per_device_eval_batch_size"],
		eval_do_concat_batches=False,
		batch_eval_metrics=True,
		no_cuda=False,
		seed=cfg.get('seed', 42),
		fp16=cfg['fp16'] if not cfg['quantized'] else False,
		fp16_backend=cfg['fp16'] if not cfg['quantized'] else False,
		disable_tqdm=True,
		gradient_checkpointing=config.training_setting['gradient_checkpointing']
		)
	
	# Initialize Trainer
	if cfg['goal'] == 'regression':
		trainer = CustomRegTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=test_dataset,
			tokenizer=None,
			data_collator=None,
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
					tokenizer=None,
					data_collator=None,
					compute_metrics=ComputeTokenClsMetricsEval())
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
					compute_metrics=ComputeMultiClsMetricsEval() if 'multi' in cfg['goal'] else ComputeClsMetricsEval())

	results = trainer.evaluate()
	return results

results = eval(cfg)
print(results)
