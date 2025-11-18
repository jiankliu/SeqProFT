import os
from datetime import datetime
from transformers import (AutoConfig, AutoTokenizer, DataCollatorWithPadding,
						  TrainingArguments)
from transformers.models.esm import EsmForDownstream
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
		'goal': cfg['goal']
	})
	
	if cfg['peft']:
		# model = get_peft_model(model, peft_config)
		model = EsmForDownstream.from_pretrained(
			ckpt_path,
			cfg['head_config']['head_name'],
			config.head_config,
			config=config).to('cuda')
		model = PeftModel.from_pretrained(model, ckpt_path)
	else:
		model = EsmForDownstream.from_pretrained(
			cfg['esm_config']['model_name'],
			cfg['head_config']['head_name'],
			config.head_config,
			config=config).to('cuda')
	sd = torch.load(os.path.join(ckpt_path, 'aux_weights.pt'), map_location=model.device)
	model.load_state_dict(sd, strict=False)
	return model, config

def prepare_dataset(cfg):
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
	tokenizer = AutoTokenizer.from_pretrained(cfg['esm_config']['model_name'])	
	if 'classification' in cfg['goal']:
		if 'GO' in cfg['dataset']['dataname'] or 'EC' in cfg['dataset']['dataname']:
			train_labels = [list(map(int, t.split(' '))) for t in train_csv.tgt_cls]
			test_labels = [list(map(int, t.split(' '))) for t in test_csv.tgt_cls]
		elif cfg['dataset']['dataname'] == 'FoldClassification' or cfg['dataset']['dataname'] == 'Loc':
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
	
	train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
	test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
	
	train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
	test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
	
	return tokenizer, train_dataset, test_dataset

def eval(cfg):
	ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	saveroot = os.path.join('results', cfg['project_name'], 'eval', f"{ts}")
	os.makedirs(saveroot, exist_ok=True)

	model, config = prepare_model(cfg, args.ckpt_path)
	tokenizer, train_dataset, test_dataset = prepare_dataset(cfg)

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
		seed=42,
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
					compute_metrics=ComputeTokenClsMetricsEval())
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
					compute_metrics=ComputeMultiClsMetricsEval() if 'multi' in cfg['goal'] else ComputeClsMetricsEval())

	results = trainer.evaluate()
	return results

results = eval(cfg)
print(results)
