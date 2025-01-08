import numpy as np
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from models.model_tester import ModelTester
from evaluation.enhanced_metrics import EnhancedMetricsEvaluator
import random
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from data.data_manager import DataManager

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Endoscopic Report Analysis System')
    
    parser.add_argument('--mode',
        choices=['evaluate', 'test', 'batch'],
        default='evaluate',
        help='Select mode: evaluate (with ground truth), test (single run), or batch (multiple models)')
    
    parser.add_argument('--models',
        nargs='+',
        choices=[
            'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro',
            'claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku', 'claude-3.5-sonnet',
            'bailian-llama3.3-70b-instruct', 'bailian-llama3.2-3b-instruct','bailian-llama3.2-1b-instruct',
            'bailian-llama3.1-405b-instruct','bailian-llama3.1-70b-instruct','bailian-llama3.1-8b-instruct',
            'meta-llama-3-405b', 'meta-llama-3.2-90b-vision', 
            'chatglm3', 
            'deepseek-chat', 
            'qwen-max', 'qwen-plus', 'qwen-turbo',
            'gpt-4', 'gpt-4o',
            'grok-beta',
            'glm-4-flash','glm-4-plus','glm-4-air'
        ],
        required=False,
        default=['gemini-2.0-flash-exp'],
        help='Select LLM model(s) to use')
    
    parser.add_argument('--split-data',
        action='store_true',
        default=False,
        help='Perform new data split')
    
    parser.add_argument('--shot-mode',
        choices=['zero', 'few'],
        default='zero',
        help='Select shot mode for evaluation')
    
    parser.add_argument('--num-shots',
        type=int,
        choices=[1, 2, 5],
        default=5,
        help='Number of shots for few-shot evaluation')
    
    parser.add_argument('--tasks',
        nargs='+',
        choices=['basic', 'relationship', 'diagnostic', 'all'],
        default=['all'],
        help='Select tasks to evaluate')
    
    parser.add_argument('--prompt-styles',
        nargs='+',
        choices=['basic_step', 'clinical_protocol', 'deductive', 'expert', 'systematic', 'all'],
        default=['all'],
        help='Select prompting styles')
    
    parser.add_argument('--config',
        type=str,
        default='config.json',
        help='Path to API configuration file')

    parser.add_argument('--ground-truth',
        default='data/ground_truth.json',
        help='Path to ground truth file (for evaluation mode)')
    
    parser.add_argument('--output-dir',
        type=str,
        default='results',
        help='Directory for output files')

    parser.add_argument('--case-ids',
        nargs='+',
        help='Specific case IDs to test (e.g., CASE_0001 CASE_0002). If not provided, all test cases will be used.')

    parser.add_argument('--data-dir',
        type=str,
        default='data',
        help='Directory containing data files')
    
    return parser.parse_args()

def filter_test_cases(test_cases: list, case_ids: list) -> list:
    """
    Filter test cases based on specified case IDs.
    
    Args:
        test_cases: List of all test cases
        case_ids: List of specific case IDs to select
        
    Returns:
        List of filtered test cases matching the specified IDs
    """
    if not case_ids:
        return test_cases
        
    filtered_cases = [case for case in test_cases if case['case_id'] in case_ids]
    
    if len(filtered_cases) != len(case_ids):
        found_ids = {case['case_id'] for case in filtered_cases}
        missing_ids = set(case_ids) - found_ids
        if missing_ids:
            logging.warning(f"Could not find the following case IDs: {', '.join(missing_ids)}")
    
    return filtered_cases

def setup_logging(args):
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'run_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        return {}

def format_case(case: dict) -> str:
    """Format case dictionary into a string for prompting."""
    return "\n".join(f"{k}: {v}" for k, v in case.items() if k not in ['case_id', 'metadata'])

def load_few_shot_examples(task: str, num_shots: int) -> str:
    """Load few-shot examples from corresponding text files."""
    try:
        example_path = Path(f"prompts/examples/{task}/{num_shots}_shot.txt")
        if not example_path.exists():
            logging.error(f"Few-shot example file not found: {example_path}")
            return ""
            
        with open(example_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error loading few-shot examples: {str(e)}")
        return ""

def run_test(tester: ModelTester, case: dict, tasks: list, styles: list, 
             shot_mode: str, num_shots: int) -> dict:
    """Run test for a single case."""
    case_results = {
        "case_id": case['case_id'],
        "tasks": {}
    }
    
    for task in tasks:
        task_results = {}
        for style in styles:
            try:
                response = tester.test_model(
                    task=task,
                    style=style,
                    case=case,
                    shot_mode=shot_mode,
                    num_shots=num_shots
                )
                task_results[style] = {
                    "success": response is not None,
                    "output": response,
                    "shot_mode": shot_mode,
                    "num_shots": num_shots if shot_mode == 'few' else 0,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logging.error(f"Error in test: {str(e)}")
                task_results[style] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        case_results["tasks"][task] = task_results
    
    return case_results

def load_test_cases(data_dir: str = "data") -> List[dict]:
    """
    Load test cases from the pre-split test_cases.json file.
    
    Args:
        data_dir: Directory containing the split data files
        
    Returns:
        List of test cases, or None if loading fails
    """
    try:
        test_file = Path(data_dir) / "test_cases.json"
        if not test_file.exists():
            logging.error("test_cases.json not found. Please ensure data has been split.")
            return None
            
        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        logging.info(f"Successfully loaded {len(test_cases)} test cases")
        return test_cases
        
    except Exception as e:
        logging.error(f"Error loading test cases: {str(e)}")
        return None

def save_results(results: dict, output_path: Path):
    """Save results to file with numpy type handling"""
    try:
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        # Convert all numpy types to native Python types
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

def print_evaluation_summary(evaluation_results: dict, model_name: str):
    """Print summary of evaluation results"""
    print(f"\nEvaluation Summary for {model_name}")
    print("=" * 50)
    
    for task, task_results in evaluation_results.get("tasks", {}).items():
        print(f"\n{task.upper()} Task:")
        print("-" * 30)
        
        for style, metrics in task_results.items():
            print(f"\nStyle: {style}")
            if isinstance(metrics, dict):
                for metric_name, score in metrics.items():
                    if isinstance(score, dict) and "mean" in score:
                        print(f"  {metric_name}:")
                        print(f"    Mean: {score['mean']:.3f} ± {score['std']:.3f}")
                        print(f"    Range: [{score['min']:.3f}, {score['max']:.3f}]")
                    elif isinstance(score, (int, float)):
                        print(f"  {metric_name}: {score:.3f}")

def evaluate_case(evaluator: EnhancedMetricsEvaluator, model_output: dict, ground_truth: dict, 
                 case_id: str, tasks: list, styles: list) -> dict:
    """Evaluate a single case"""
    evaluation_results = {
        "case_id": case_id,
        "tasks": {}
    }
    
    for task in tasks:
        try:
            evaluation_results["tasks"][task] = {}
            ground_truth_task = ground_truth.get(case_id, {}).get("tasks", {}).get(task, {})
            
            if not ground_truth_task:
                logging.warning(f"Missing ground truth for task {task} in case {case_id}")
                continue

            for style in styles:
                try:
                    style_output = model_output.get("tasks", {}).get(task, {}).get(style, {})
                    if not style_output or not style_output.get("success"):
                        continue

                    prediction = style_output.get("output", {})
                    
                    if task == "basic":
                        result = evaluator.evaluate_basic_task(prediction, ground_truth_task)
                    elif task == "relationship":
                        result = evaluator.evaluate_relationship_task(prediction, ground_truth_task)
                    elif task == "diagnostic":
                        result = evaluator.evaluate_diagnostic_task(prediction, ground_truth_task)
                    
                    evaluation_results["tasks"][task][style] = result
                    
                except Exception as e:
                    logging.error(f"Error evaluating {task} for style {style} in case {case_id}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error evaluating task {task} for case {case_id}: {str(e)}")
    
    return evaluation_results


def main():
    args = parse_args()
    setup_logging(args)
    
    # Initialize data manager
    data_manager = DataManager()
    if not data_manager.load_data():
        logging.error("Failed to load data")
        return
    
    if args.split_data:
        test_cases, validation_cases, few_shot_cases = data_manager.split_data()
        if not all([test_cases, validation_cases, few_shot_cases]):
            logging.error("Failed to split data")
            return
        if not data_manager.save_splits():
            logging.error("Failed to save data splits")
            return
    else:
        test_cases = load_test_cases(args.data_dir)

    if args.case_ids:
        selected_cases = [case for case in test_cases if case['case_id'] in args.case_ids]
        if not selected_cases:
            logging.error("None of the specified case IDs were found in the test set")
            return
        logging.info(f"Selected {len(selected_cases)} cases for evaluation")
    else:
        selected_cases = test_cases
        logging.info(f"Using all {len(selected_cases)} test cases")
    
    # Prepare tasks and styles
    tasks = ['basic', 'relationship', 'diagnostic'] if 'all' in args.tasks else args.tasks
    styles = ['basic_step', 'clinical_protocol', 'deductive', 'expert', 'systematic'] if 'all' in args.prompt_styles else args.prompt_styles

    # Process each model
    for model_name in args.models:
        logging.info(f"Processing model: {model_name}")
        
        try:
            tester = ModelTester(args.config, model_name)
            
            # Run tests
            if args.mode in ['test', 'batch']:
                results = {
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "mode": args.mode,
                        "tasks": tasks,
                        "styles": styles,
                        "shot_mode": args.shot_mode,
                        "num_shots": args.num_shots if args.shot_mode == 'few' else 0,
                        "selected_cases": args.case_ids if args.case_ids else "all"
                    },
                    "results": []
                }
           
                for case in selected_cases:
                    case_result = run_test(
                        tester=tester,
                        case=case,
                        tasks=tasks,
                        styles=styles,
                        shot_mode=args.shot_mode,
                        num_shots=args.num_shots
                    )
                    results["results"].append(case_result)
                
                # Save results
                task_str = '_'.join(args.tasks) if isinstance(args.tasks, list) else args.tasks
                output_file = Path(args.output_dir) / f'test_{model_name}_{task_str}_{args.shot_mode}_{args.num_shots}.json'
                save_results(results, output_file)

            if args.mode in ['evaluate', 'batch']:
                evaluator = EnhancedMetricsEvaluator()

                try:
                    with open(args.ground_truth, 'r') as f:
                        ground_truth = json.load(f)
                except Exception as e:
                    logging.error(f"Error loading ground truth: {str(e)}")
                    continue
                
                # Load test results if they exist
                task_str = '_'.join(args.tasks) if isinstance(args.tasks, list) else args.tasks
                results_file = Path(args.output_dir) / f'test_{model_name}_{task_str}_{args.shot_mode}_{args.num_shots}.json'
                try:
                    with open(results_file, 'r') as f:
                        test_results = json.load(f)
                except FileNotFoundError:
                    if args.mode == 'evaluate':
                        logging.error(f"No test results found for {model_name}")
                        continue
                    test_results = results
                
                # Run evaluation
                evaluation_results = {
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "evaluations": []
                }
                
                for case_result in test_results["results"]:
                    case_id = case_result["case_id"]
                    if case_id in ground_truth:
                        evaluation = evaluate_case(
                            evaluator=evaluator,
                            model_output=case_result,
                            ground_truth=ground_truth,
                            case_id=case_id,
                            tasks=tasks,
                            styles=styles
                        )
                        evaluation_results["evaluations"].append(evaluation)
                
                # Calculate aggregate metrics
                evaluation_results["aggregate_metrics"] = evaluator.calculate_aggregate_metrics(
                    evaluation_results["evaluations"]
                )
                
                # Save evaluation results
                output_file = Path(args.output_dir) / f'evaluation_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                save_results(evaluation_results, output_file)
                
                # Print summary
                print_evaluation_summary(evaluation_results, model_name)
        
        except Exception as e:
            logging.error(f"Error processing model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()