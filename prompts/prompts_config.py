# prompts/prompts_config.py

from pathlib import Path
import json
import logging

class PromptsManager:
    def __init__(self):
        self.prompt_dir = Path("prompts/templates")
        self.examples_dir = Path("prompts/examples")
        self.task_dirs = ['basic', 'relationship', 'diagnostic']
        self.prompt_types = [
            'basic_step_cot.txt',
            'clinical_protocol.txt',
            'deductive_cot.txt',
            'expert_cot.txt',
            'systematic_cot.txt',
            'format.txt'
        ]
        
        # Verify directory structure
        self._initialize_directory_structure()

    def _initialize_directory_structure(self):
        """Initialize and verify the prompt directory structure"""
        try:
            # Initialize templates directory
            self.prompt_dir.mkdir(parents=True, exist_ok=True)
            for task_dir in self.task_dirs:
                task_path = self.prompt_dir / task_dir
                task_path.mkdir(exist_ok=True)
                
                # Verify all required prompt files exist
                for prompt_type in self.prompt_types:
                    prompt_file = task_path / prompt_type
                    if not prompt_file.exists():
                        logging.warning(f"Missing prompt file: {prompt_file}")
            
            # Initialize examples directory
            self.examples_dir.mkdir(parents=True, exist_ok=True)
            for task_dir in self.task_dirs:
                example_path = self.examples_dir / task_dir
                example_path.mkdir(exist_ok=True)
                
        except Exception as e:
            logging.error(f"Error initializing directory structure: {str(e)}")

    def load_prompt(self, task: str, style: str, shot_mode: str = 'zero', num_shots: int = 0) -> str:
        """
        Load prompt template with format definition and examples if specified.
        
        Args:
            task: One of 'basic', 'relationship', 'diagnostic'
            style: One of 'basic_step', 'clinical_protocol', 'deductive', 'expert', 'systematic'
            shot_mode: 'zero' or 'few' for shot configuration
            num_shots: Number of shots (1, 2, or 5) when using few-shot mode
        """
        try:
            task_dir = self.prompt_dir / task
            
            # Load format file
            format_file = task_dir / 'format.txt'
            with open(format_file, 'r') as f:
                format_content = f.read()

            # Map style to filename
            style_mapping = {
                'basic_step': 'basic_step_cot.txt',
                'clinical_protocol': 'clinical_protocol.txt',
                'deductive': 'deductive_cot.txt',
                'expert': 'expert_cot.txt',
                'systematic': 'systematic_cot.txt'
            }
            
            template_file = task_dir / style_mapping[style]
            
            # Load specific prompt template
            with open(template_file, 'r') as f:
                template_content = f.read()

            # Replace format inclusion directive
            full_prompt = template_content.replace(
                f"[Include content of {task}_format.txt at the start of this file]",
                format_content
            )
            
            # Add few-shot examples if specified
            if shot_mode == 'few' and num_shots > 0:
                example_file = self.examples_dir / task / f"{num_shots}_shot.txt"
                if example_file.exists():
                    with open(example_file, 'r') as f:
                        examples = f.read().strip()
                    full_prompt = f"{examples}\n\n{full_prompt}"
                else:
                    logging.warning(f"Few-shot example file not found: {example_file}")
            
            return full_prompt
            
        except KeyError:
            logging.error(f"Invalid style specified: {style}")
            return None
        except FileNotFoundError as e:
            logging.error(f"Prompt file not found: {e.filename}")
            return None
        except Exception as e:
            logging.error(f"Error loading prompt: {str(e)}")
            return None

    def format_prompt(self, template: str, case_data: dict) -> str:
        """Format prompt template with case data"""
        if template is None:
            return None
            
        # Convert case data to formatted string, excluding metadata
        case_str = "\n".join([
            f"{k}: {v}" 
            for k, v in case_data.items() 
            if k not in ["case_id", "metadata"]
        ])
        
        return template.replace("{case_data}", case_str)