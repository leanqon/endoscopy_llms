# advanced_metrics.py
from typing import Dict, List
import numpy as np
import logging
from collections import defaultdict

class EnhancedMetricsEvaluator:
    def evaluate_basic_task(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Basic entity extraction task evaluation"""
        try:
            pred_findings = set((f['location'], l['type']) 
                              for f in prediction.get('findings', [])
                              for l in f.get('lesions', []))
            true_findings = set((f['location'], l['type']) 
                              for f in ground_truth.get('findings', [])
                              for l in f.get('lesions', []))
            
            tp = len(pred_findings & true_findings)
            fp = len(pred_findings - true_findings)
            fn = len(true_findings - pred_findings)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except Exception as e:
            logging.error(f"Error in basic task evaluation: {str(e)}")
            return {"precision": 0, "recall": 0, "f1": 0}

    def evaluate_relationship_task(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Pattern recognition task evaluation"""
        try:
            # Pattern accuracy
            pred_patterns = set(p['finding_type'] for p in prediction.get('patterns', []))
            true_patterns = set(p['finding_type'] for p in ground_truth.get('patterns', []))
            pattern_acc = len(pred_patterns & true_patterns) / len(true_patterns) if true_patterns else 0
            
            # Location accuracy
            pred_locs = set(loc for p in prediction.get('patterns', []) 
                          for loc in p.get('locations', []))
            true_locs = set(loc for p in ground_truth.get('patterns', []) 
                          for loc in p.get('locations', []))
            location_acc = len(pred_locs & true_locs) / len(true_locs) if true_locs else 0
            
            # Type accuracy
            pred_types = set(p.get('distribution') for p in prediction.get('patterns', []))
            true_types = set(p.get('distribution') for p in ground_truth.get('patterns', []))
            type_acc = len(pred_types & true_types) / len(true_types) if true_types else 0
            
            return {
                "pattern_accuracy": pattern_acc,
                "location_accuracy": location_acc,
                "type_accuracy": type_acc
            }
        except Exception as e:
            logging.error(f"Error in relationship task evaluation: {str(e)}")
            return {"pattern_accuracy": 0, "location_accuracy": 0, "type_accuracy": 0}

    def evaluate_diagnostic_task(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Diagnostic assessment task evaluation"""
        try:
            # Diagnosis accuracy
            pred_diag = set(prediction.get('suggested_diagnoses', []))
            true_diag = set(ground_truth.get('suggested_diagnoses', []))
            diag_acc = len(pred_diag & true_diag) / len(true_diag) if true_diag else 0
            
            # Classification accuracy
            class_acc = 1.0 if (prediction.get('classification') == 
                              ground_truth.get('classification')) else 0.0
            
            return {
                "diagnosis_accuracy": diag_acc,
                "classification_accuracy": class_acc
            }
        except Exception as e:
            logging.error(f"Error in diagnostic task evaluation: {str(e)}")
            return {"diagnosis_accuracy": 0, "classification_accuracy": 0}

    def calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict:
        """Calculate metrics across test cases by prompting style"""
        aggregate_metrics = defaultdict(
            lambda: {
                "basic": defaultdict(list),
                "relationship": defaultdict(list),
                "diagnostic": defaultdict(list)
            }
        )

        for eval_result in evaluations:
            for task, task_results in eval_result.get("tasks", {}).items():
                for style, metrics in task_results.items():
                    if task == "basic":
                        aggregate_metrics[style]["basic"]["precision"].append(metrics.get("precision", 0))
                        aggregate_metrics[style]["basic"]["recall"].append(metrics.get("recall", 0))
                        aggregate_metrics[style]["basic"]["f1"].append(metrics.get("f1", 0))
                    elif task == "relationship":
                        aggregate_metrics[style]["relationship"]["pattern_accuracy"].append(metrics.get("pattern_accuracy", 0))
                        aggregate_metrics[style]["relationship"]["location_accuracy"].append(metrics.get("location_accuracy", 0))
                        aggregate_metrics[style]["relationship"]["type_accuracy"].append(metrics.get("type_accuracy", 0))
                    elif task == "diagnostic":
                        aggregate_metrics[style]["diagnostic"]["diagnosis_accuracy"].append(metrics.get("diagnosis_accuracy", 0))
                        aggregate_metrics[style]["diagnostic"]["classification_accuracy"].append(metrics.get("classification_accuracy", 0))

        final_metrics = {}
        for style in aggregate_metrics:
            final_metrics[style] = {
                task: {
                    metric: float(np.mean(values))
                    for metric, values in metrics.items() if values
                }
                for task, metrics in aggregate_metrics[style].items()
            }

        return final_metrics

    def calculate_aggregate_metrics_old(self, evaluations: List[Dict]) -> Dict:
        """Calculate final metrics across all test cases"""
        task_metrics = {
            "basic": {
                "precision": [], "recall": [], "f1": []
            },
            "relationship": {
                "pattern_accuracy": [], "location_accuracy": [], "type_accuracy": []
            },
            "diagnostic": {
                "diagnosis_accuracy": [], "classification_accuracy": []
            }
        }

        for eval_result in evaluations:
            for task, task_results in eval_result.get("tasks", {}).items():
                for style, metrics in task_results.items():
                    if task == "basic":
                        task_metrics["basic"]["precision"].append(metrics.get("precision", 0))
                        task_metrics["basic"]["recall"].append(metrics.get("recall", 0))
                        task_metrics["basic"]["f1"].append(metrics.get("f1", 0))
                    elif task == "relationship":
                        task_metrics["relationship"]["pattern_accuracy"].append(
                            metrics.get("pattern_accuracy", 0))
                        task_metrics["relationship"]["location_accuracy"].append(
                            metrics.get("location_accuracy", 0))
                        task_metrics["relationship"]["type_accuracy"].append(
                            metrics.get("type_accuracy", 0))
                    elif task == "diagnostic":
                        task_metrics["diagnostic"]["diagnosis_accuracy"].append(
                            metrics.get("diagnosis_accuracy", 0))
                        task_metrics["diagnostic"]["classification_accuracy"].append(
                            metrics.get("classification_accuracy", 0))

        return {
            task: {
                metric: float(np.mean(values)) 
                for metric, values in metrics.items() if values
            }
            for task, metrics in task_metrics.items()
        }