import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer
from datasets import Dataset
from accelerate import Accelerator
from peft import PeftModel
from scipy.stats import spearmanr, pearsonr
from sklearn import metrics

# Helper functions and data preparation
def truncate_labels(labels, max_length):
	"""Truncate labels to the specified max_length."""
	return [label[:max_length] for label in labels]

def compute_metrics(p):
	"""Compute metrics for evaluation."""
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	# Remove padding (-100 labels)
	predictions = predictions[labels != -100].flatten()
	labels = labels[labels != -100].flatten()

	# Compute accuracy
	accuracy = accuracy_score(labels, predictions)

	# Compute precision, recall, F1 score, and AUC
	precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
	auc = roc_auc_score(labels, predictions)

	# Compute MCC
	mcc = matthews_corrcoef(labels, predictions)

	return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

# torchdrug
def area_under_roc(pred, target):
	"""
	Area under receiver operating characteristic curve (ROC).

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): binary targets of shape :math:`(n,)`
	"""
	order = pred.argsort(descending=True)
	target = target[order]
	hit = target.cumsum(0)
	all = (target == 0).sum() * (target == 1).sum()
	auroc = hit[target == 0].sum() / (all + 1e-10)
	return auroc

def area_under_prc(pred, target):
	"""
	Area under precision-recall curve (PRC).

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): binary targets of shape :math:`(n,)`
	"""
	order = pred.argsort(descending=True)
	target = target[order]
	precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
	auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
	return auprc

# torchdrug
def f1_max(pred, target):
	"""
	F1 score with the optimal threshold.

	This function first enumerates all possible thresholds for deciding positive and negative
	samples, and then pick the threshold with the maximal F1 score.

	Parameters:
		pred (Tensor): predictions of shape :math:`(B, N)`
		target (Tensor): binary targets of shape :math:`(B, N)`
	"""
	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)

	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
					torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
				 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1.max()

def accuracy(pred, target):
	return (pred.argmax(dim=-1) == target).float().mean()

def compute_cls_metrics(p):
	# preds: (num_samples, num_cls), fp32
	# labels: (num_samples, num_cls), fp32
	preds, labels = p
	#preds = preds.reshape(-1)
	#labels = labels.reshape(-1)
	preds = torch.tensor(preds)
	labels = torch.tensor(labels)
	auprc_micro = area_under_prc(preds.flatten(), labels.long().flatten())
	f1max = f1_max(preds, labels)
	return {'auprc@micro': auprc_micro, 'f1_max': f1max}


class ComputeClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			result = {
				'auprc@micro': area_under_prc(torch.stack(self.batch_preds).flatten(), torch.stack(self.batch_labels).long().flatten()),
				'f1_max': f1_max(torch.stack(self.batch_preds), torch.stack(self.batch_labels))}
			self.batch_preds, self.batch_labels = [], []
			return result

##################################
# add sensitivity, specificity, precision for rebuttal
from sklearn.metrics import confusion_matrix

def calculate_binary_metrics(y_pred, y_true):
    """Calculate sensitivity, specificity, precision for binary classification."""
    # Convert to numpy for sklearn compatibility
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return sensitivity, specificity, precision

class ComputeClsMetricsEval:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			predictions = torch.stack(self.batch_preds)
			gts = torch.stack(self.batch_labels)

			result = {
				'auprc@micro': area_under_prc(predictions.flatten(), gts.long().flatten()),
				'f1_max': f1_max(predictions, gts)}
			binary_preds = (predictions > 0.5).float()
			num_classes = predictions.shape[1]
			macro_sensitivity, macro_specificity, macro_precision = [], [], []
			for i in range(num_classes):
				class_preds = binary_preds[:, i]
				class_gts = gts[:, i]
				if torch.sum(class_gts) == 0:
					continue
				sens, spec, prec = calculate_binary_metrics(class_preds, class_gts)
				macro_sensitivity.append(sens)
				macro_specificity.append(spec)
				macro_precision.append(prec)
			result['macro_sensitivity'] = np.mean(macro_sensitivity) if macro_sensitivity else 0
			result['macro_specificity'] = np.mean(macro_specificity) if macro_specificity else 0
			result['macro_precision'] = np.mean(macro_precision) if macro_precision else 0

			flat_preds = binary_preds.flatten().numpy()
			flat_gts = gts.flatten().numpy()

			micro_sens, micro_spec, micro_prec = calculate_binary_metrics(flat_preds, flat_gts)
			result['micro_sensitivity'] = micro_sens
			result['micro_specificity'] = micro_spec
			result['micro_precision'] = micro_prec

			self.batch_preds, self.batch_labels = [], []
			return result


class ComputeMultiClsMetricsEval:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			logits = torch.stack(self.batch_preds)
			gts = torch.stack(self.batch_labels).long()

			predicted_classes = torch.argmax(logits, dim=1)
			result = {'accuracy': accuracy(logits, gts)}

			num_classes = logits.shape[1]
			gt_onehot = torch.zeros(gts.size(0), num_classes)
			gt_onehot.scatter_(1, gts.unsqueeze(1), 1)
			pred_onehot = torch.zeros(predicted_classes.size(0), num_classes)
			pred_onehot.scatter_(1, predicted_classes.unsqueeze(1), 1)

			sens, spec, prec = [], [], []

			for i in range(num_classes):
				class_preds = pred_onehot[:, i]
				class_gts = gt_onehot[:, i]
				if torch.sum(class_gts) == 0:
					continue
				sens_i, spec_i, prec_i = calculate_binary_metrics(class_preds, class_gts)
				sens.append(sens_i)
				spec.append(spec_i)
				prec.append(prec_i)
			
			result['macro_sensitivity'] = np.mean(sens) if sens else 0
			result['macro_specificity'] = np.mean(spec) if spec else 0
			result['macro_precision'] = np.mean(prec) if prec else 0

			all_preds_flat = pred_onehot.reshape(-1).numpy()
			all_gts_flat = gt_onehot.reshape(-1).numpy()
			micro_sens, micro_spec, micro_prec = calculate_binary_metrics(all_preds_flat, all_gts_flat)
			result['micro_sensitivity'] = micro_sens
			result['micro_specificity'] = micro_spec
			result['micro_precision'] = micro_prec

			self.batch_preds, self.batch_labels = [], []
			return result


class ComputeTokenClsMetricsEval:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]

		batch_size, seq_len, num_cls = preds.shape
		assert batch_size == 1
		out_label_list = [[] for _ in range(batch_size)]
		preds_list = [[] for _ in range(batch_size)]
		for i in range(batch_size):
			for j in range(seq_len):
				if labels[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
					out_label_list[i].append(labels[i][j].detach().cpu())
					preds_list[i].append(preds[i][j].detach().cpu())
		self.batch_preds.extend(preds_list[0])
		self.batch_labels.extend(out_label_list[0])
		if compute_result:
			result = {
				'accuracy': accuracy(torch.stack(self.batch_preds), torch.stack(self.batch_labels).long()),
			}
			pred_tensors = torch.stack(self.batch_preds)
			pred_classes = torch.argmax(pred_tensors, dim=1).numpy()
			gt_classes = torch.stack(self.batch_labels).long().numpy()
			sens, spec, prec = [], [], []
			unique_classes = np.unique(gt_classes)
			num_classes = len(unique_classes)
			pred_onehot = np.zeros((len(pred_classes), num_classes))
			gt_onehot = np.zeros((len(gt_classes), num_classes))			
			for i, cls_ in enumerate(unique_classes):
				binary_preds = (pred_classes == cls_).astype(int)
				binary_gts = (gt_classes == cls_).astype(int)
				tn, fp, fn, tp = confusion_matrix(binary_gts, binary_preds).ravel()
				sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0
				spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0
				prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0
				sens.append(sens_i)
				spec.append(spec_i)
				prec.append(prec_i)
				result[f'class_{cls_}_sensitivity'] = sens_i
				result[f'class_{cls_}_specificity'] = spec_i
				result[f'class_{cls_}_precision'] = prec_i
				gt_onehot[gt_classes == cls_, i] = 1
				pred_onehot[pred_classes == cls_, i] = 1
			
			result['macro_sensitivity'] = np.mean(sens)
			result['macro_specificity'] = np.mean(spec)
			result['macro_precision'] = np.mean(prec)

			tn, fp, fn, tp = confusion_matrix(gt_onehot.flatten(), pred_onehot.flatten()).ravel()
			micro_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
			micro_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
			micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
			result['micro_sensitivity'] = micro_sens
			result['micro_specificity'] = micro_spec
			result['micro_precision'] = micro_prec
			
			self.batch_preds, self.batch_labels = [], []
			return result
##################################

class ComputeMultiClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			result = {
				'accuracy': accuracy(torch.stack(self.batch_preds), torch.stack(self.batch_labels).long())}
			self.batch_preds, self.batch_labels = [], []
			return result

def compute_reg_metrics(p):
	predictions, labels = p
	predictions = predictions.reshape(-1)
	labels = labels.reshape(-1)
	spearman = spearmanr(predictions, labels)[0]
	pearson = pearsonr(predictions, labels)[0]
	rmse = np.sqrt(metrics.mean_squared_error(predictions, labels))
	r2 = metrics.r2_score(labels, predictions)
	return {'spearman': spearman, 'pearson': pearson, 'rmse': rmse, 'r2': r2}

class ComputeTokenClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		# preds = preds[:, 1:-1]
		# preds = np.argmax(preds, axis=2)
		batch_size, seq_len, num_cls = preds.shape
		assert batch_size == 1
		out_label_list = [[] for _ in range(batch_size)]
		preds_list = [[] for _ in range(batch_size)]
		for i in range(batch_size):
			for j in range(seq_len):
				if labels[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
					out_label_list[i].append(labels[i][j].detach().cpu())
					preds_list[i].append(preds[i][j].detach().cpu())
		self.batch_preds.extend(preds_list[0])
		self.batch_labels.extend(out_label_list[0])
		if compute_result:
			result = {
				'accuracy': accuracy(torch.stack(self.batch_preds), torch.stack(self.batch_labels).long()),
			}
			self.batch_preds, self.batch_labels = [], []
			return result


class ComputeRegMetrics:
	def __init__(self):
		self.batch_preds = []
		self.batch_labels = []
	
	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		preds = preds.reshape(-1).cpu().numpy()
		labels = labels.reshape(-1).cpu().numpy()

		self.batch_preds.extend(preds)
		self.batch_labels.extend(labels)
     
		if compute_result:
			result = {'spearman': spearmanr(self.batch_preds, self.batch_labels)[0],
					  'pearson': pearsonr(self.batch_preds, self.batch_labels)[0],
					  'rmse': np.sqrt(metrics.mean_squared_error(self.batch_preds, self.batch_labels)),
					  'r2': metrics.r2_score(self.batch_labels, self.batch_preds)}
			self.batch_preds = []
			self.batch_labels = []
			return result
		

class WeightedTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		"""Custom compute_loss function."""
		outputs = model(**inputs)
		loss_fct = nn.CrossEntropyLoss()
		active_loss = inputs["attention_mask"].view(-1) == 1
		active_logits = outputs.logits.view(-1, model.config.num_labels)
		active_labels = torch.where(
			active_loss, inputs["labels"].view(-1), torch.tensor(loss_fct.ignore_index).type_as(inputs["labels"])
		)
		loss = loss_fct(active_logits, active_labels)
		return (loss, outputs) if return_outputs else loss

