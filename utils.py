import torch
import torch.nn.functional as F
from transformers import Trainer
import json
import torch.nn as nn
import os
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_peft_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import safetensors.torch

if is_peft_available():
    from peft import PeftModel

# Helper Functions and Data Preparation
def save_config_to_txt(config, filename):
	"""Save the configuration dictionary to a text file."""
	with open(filename, 'w') as f:
		json.dump(config, f)
 
# print trainable params
def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"[Params] trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)


class CustomRegTrainer(Trainer):
	def __init__(self, aux_list=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.aux_list = aux_list

	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.pop('labels')
		outputs = model(**inputs)
		logits = outputs.logits
		loss = nn.MSELoss()(logits.view(-1), labels.view(-1))
		if torch.isnan(loss):
			del model
			torch.cuda.empty_cache()			
			raise Exception(f'Loss is Nan')
		torch.cuda.empty_cache()
		return (loss, outputs) if return_outputs else loss
	
	def _save(self, output_dir: Optional[str] = None, state_dict=None):
		# If we are executing this function, we are the process zero, so we don't check for that.
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		print(f"Saving model checkpoint to {output_dir}")

		supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
		# Save a trained model and configuration using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		if not isinstance(self.model, supported_classes):
			if state_dict is None:
				state_dict = self.model.state_dict()

			if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
				self.accelerator.unwrap_model(self.model).save_pretrained(
					output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
				)
			else:
				print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
				if self.args.save_safetensors:
					safetensors.torch.save_file(
						state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
					)
				else:
					torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
		else:
			self.model.save_pretrained(
				output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
			)

		if self.tokenizer is not None:
			self.tokenizer.save_pretrained(output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

		# save auxiliary weights
		if self.aux_list is not None:
			tensors = dict()
			for aux_name in self.aux_list:
				for k, v in self.model.named_parameters():
					if aux_name in k:
						tensors[k] = v.detach()
			torch.save(tensors, os.path.join(output_dir, 'aux_weights.pt'))


class CustomClsTrainer(Trainer):
	def __init__(self, aux_list=None, criterion='bce', *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.aux_list = aux_list
		self.criterion = criterion
			
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.pop('labels')
		outputs = model(**inputs)
		logits = outputs.logits
		# labels: (bs, num_cls), float32
		# logits: (bs, num_cls), float32
		if self.criterion == 'bce':
			loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
			loss = loss.mean(dim=0).sum()
		elif self.criterion == 'ce':
			if logits.ndim == 3:
				num_classes = logits.shape[-1]
				loss = F.cross_entropy(logits.view(-1, num_classes), labels.view(-1).long(), reduction='mean')
			else:
				loss = F.cross_entropy(logits, labels.long(), reduction='mean')
		elif 'bcew' in self.criterion:  # cross entropy with weight
			pos_weight = torch.tensor(float(self.criterion.split('_')[1])).to(logits.device)
			loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float(), reduction='mean', pos_weight=pos_weight)
		if torch.isnan(loss):
			del model
			torch.cuda.empty_cache()
			raise Exception(f'Loss is Nan')
		torch.cuda.empty_cache()
		return (loss, outputs) if return_outputs else loss

	def _save(self, output_dir: Optional[str] = None, state_dict=None):
		# If we are executing this function, we are the process zero, so we don't check for that.
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		print(f"Saving model checkpoint to {output_dir}")

		supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
		# Save a trained model and configuration using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		if not isinstance(self.model, supported_classes):
			if state_dict is None:
				state_dict = self.model.state_dict()

			if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
				self.accelerator.unwrap_model(self.model).save_pretrained(
					output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
				)
			else:
				print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
				if self.args.save_safetensors:
					safetensors.torch.save_file(
						state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
					)
				else:
					torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
		elif isinstance(self.model, PeftModel):
			self.model.save_pretrained(
				output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
			)
		else:
			pass

		if self.tokenizer is not None:
			self.tokenizer.save_pretrained(output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

		# save auxiliary weights
		if self.aux_list is not None:
			tensors = dict()
			for aux_name in self.aux_list:
				for k, v in self.model.named_parameters():
					if aux_name in k:
						tensors[k] = v.detach()
			torch.save(tensors, os.path.join(output_dir, 'aux_weights.pt'))	
