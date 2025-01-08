from typing import Dict, List
import numpy as np

class Evaluator:
    def __init__(self, golden_standards: List[Dict]):
        self.golden_standards = golden_standards
        
    def evaluate(self, result: Dict, standard: Dict, task: str) -> Dict:
        if task == "basic_extraction":
            return self._evaluate_extraction(result, standard)
        elif task == "pathology_analysis":
            return self._evaluate_pathology(result, standard)
        elif task == "diagnostic_reasoning":
            return self._evaluate_diagnosis(result, standard)
            
    def _evaluate_extraction(self, result: Dict, standard: Dict) -> Dict:
        metrics = {
            "findings_accuracy": self._compare_findings(
                result.get("findings", []),
                standard.get("findings", [])
            )
        }
        return metrics
        
    def _evaluate_pathology(self, result: Dict, standard: Dict) -> Dict:
        metrics = {
            "pathology_accuracy": self._compare_pathology(
                result.get("pathology", []),
                standard.get("pathology", [])
            )
        }
        return metrics
        
    def _evaluate_diagnosis(self, result: Dict, standard: Dict) -> Dict:
        metrics = {
            "diagnosis_accuracy": self._compare_diagnoses(
                result.get("final_diagnosis", []),
                standard.get("final_diagnosis", [])
            ),
            "classification_accuracy": 
                result.get("classification") == standard.get("classification")
        }
        return metrics
