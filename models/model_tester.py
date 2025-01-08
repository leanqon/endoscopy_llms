from google.auth.transport.requests import Request
from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import anthropic
import openai
from openai import OpenAI
import requests
from typing import Dict, List
import json
import logging
from prompts.prompts_config import PromptsManager
import time

class ModelTester:
    MODEL_MAPPINGS = {
        # Vertex AI models
        "meta-llama-3-405b": "meta/llama-3.1-405b-instruct-maas",
        #"llama-3.2-90b": "meta/llama-3.2-90b-instruct-maas",
        "meta-llama-3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct-maas",
        
        # Llama models via Bailian
        "bailian-llama3.3-70b-instruct": "llama3.3-70b-instruct",
        "bailian-llama3.2-3b-instruct": "llama3.2-3b-instruct",
        "bailian-llama3.2-1b-instruct": "llama3.2-1b-instruct",
        "bailian-llama3.1-405b-instruct": "llama3.1-405b-instruct",
        "bailian-llama3.1-70b-instruct": "llama3.1-70b-instruct",
        "bailian-llama3.1-8b-instruct": "llama3.1-8b-instruct",
        
        # Claude models - Vertex AI identifiers
        "claude-3-sonnet-vertex": "anthropic/claude-3-sonnet@latest",
        "claude-3-opus-vertex": "anthropic/claude-3-opus@latest",
        "claude-3-haiku-vertex": "anthropic/claude-3-haiku@latest",
        "claude-3.5-sonnet-vertex": "anthropic/claude-3.5-sonnet-v2@latest",
        
        # Claude models - Direct API identifiers
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        
        # Grok model - Direct mapping for Grok
        "grok-beta": "grok-beta" ,

        # GLM series models
        "glm-4-plus": "glm-4-plus",
        "glm-4-flash": "glm-4-flash",
        "glm-4-air": "glm-4-air",

         # DeepSeek models
        "deepseek-chat": "deepseek-chat",

        # Qwen models
        "qwen-max": "qwen-max-2024-09-19",
        "qwen-plus": "qwen-plus-2024-12-20",
        "qwen-turbo": "qwen-turbo-2024-11-01"
    }

    def __init__(self, config_path: str, model_name: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.model_name = model_name
        self.prompt_manager = PromptsManager()
        self.setup_client()

    def setup_client(self):
        try:
            if "gemini" in self.model_name:
                self._setup_gemini()
            elif "claude" in self.model_name:
                self._setup_claude()
            elif "gpt" in self.model_name:
                self._setup_openai()
            elif "meta-llama" in self.model_name:
                self._setup_vertex_llama()
            elif "grok" in self.model_name:
                self._setup_grok()
            elif "glm" in self.model_name:
                self._setup_glm()
            elif "deepseek" in self.model_name:
                self._setup_deepseek() 
            elif any(model in self.model_name for model in ["llama3.1", "llama3.2", "llama3.3"]):
                self._setup_bailian()
            elif "qwen" in self.model_name:
                self._setup_qwen()
            logging.info(f"Successfully set up {self.model_name} client")
        except Exception as e:
            logging.error(f"Error setting up client: {str(e)}")
            raise

    def _setup_deepseek(self):
        """Setup DeepSeek API client"""
        if "deepseek_api_key" not in self.config["api_keys"]:
            raise ValueError("DeepSeek API key not found in config")
        self.deepseek_api_key = self.config["api_keys"]["deepseek_api_key"]
        self.deepseek_api_url = self.config.get(
            "deepseek_api_url", 
            "https://api.deepseek.com/v1/chat/completions"
        )

    def _setup_glm(self):
        """Setup GLM API client with specific model configurations"""
        if "glm_api_key" not in self.config["api_keys"]:
            raise ValueError("GLM API key not found in config")
            
        self.glm_api_key = self.config["api_keys"]["glm_api_key"]
        self.glm_api_url = self.config.get("glm_api_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        
        # Model-specific configurations
        self.glm_configs = {
            "glm-4-plus": {
                "max_tokens": 16384,
                "temperature": 0.3,
                "top_p": 0.7
            },
            "glm-4-flash": {
                "max_tokens": 4096,
                "temperature": 0.3,
                "top_p": 0.7,
                "support_vision": True
            },
            "glm-4-air": {
                "max_tokens": 4096,
                "temperature": 0.3,
                "top_p": 0.7
            }
        }

    def _setup_grok(self):
        """Setup Grok API client"""
        if "xai_api_key" not in self.config["api_keys"]:
            raise ValueError("X.AI API key not found in config")
        self.xai_api_key = self.config["api_keys"]["xai_api_key"]

    def _setup_gemini(self):
        """Setup Gemini API client with rate limiting"""
        vertexai.init(
            project=self.config["google"]["project_id"],
            location=self.config["google"]["location"]
        )
        
        # Model-specific rate limits (requests per minute)
        self.rate_limits = {
            "gemini-1.5-flash": 1,
            "gemini-experimental": 2
        }
        
        # Initialize rate limiting state
        self.last_request_time = {}
        
        model_version = self.model_name
        self.model = GenerativeModel(model_version)
        self.chat = self.model.start_chat()

    def _enforce_rate_limit(self):
        """Enforce rate limiting for Gemini models"""
        current_time = time.time()
        
        # Get rate limit for current model
        rate_limit = self.rate_limits.get(self.model_name, 1)  # Default to 1 request per minute
        min_interval = 60.0 / rate_limit  # Minimum time between requests in seconds
        
        # Initialize last request time if not exists
        if self.model_name not in self.last_request_time:
            self.last_request_time[self.model_name] = 0
            
        # Calculate time since last request
        time_since_last = current_time - self.last_request_time[self.model_name]
        
        # If needed, sleep to maintain rate limit
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds for {self.model_name}")
            time.sleep(sleep_time)
            
        # Update last request time
        self.last_request_time[self.model_name] = time.time()

    def _setup_claude(self):
        if self.config.get("use_vertex_api", False):
            self.endpoint = f"https://{self.config['vertex']['location']}-aiplatform.googleapis.com"
            self.token = self._get_vertex_token()
        else:
            self.client = anthropic.Anthropic(api_key=self.config["api_keys"]["anthropic"])

    def _setup_openai(self):
        self.client = openai.Client(api_key=self.config["api_keys"]["openai"])

    def _setup_vertex_llama(self):
        self.endpoint = f"https://{self.config['vertex']['location']}-aiplatform.googleapis.com"
        self.token = self._get_vertex_token()

    def _setup_bailian(self):
        """Setup Aliyun Bailian API client"""
        if "bailian_api_key" not in self.config["api_keys"]:
            raise ValueError("Bailian API key not found in config")
            
        self.bailian_api_key = self.config["api_keys"]["bailian_api_key"]
        self.bailian_api_url = self.config.get(
            "bailian_api_url", 
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        )

    def _setup_qwen(self):
        """Setup Qwen API client using OpenAI compatibility mode"""
        if "bailian_api_key" not in self.config["api_keys"]:
            raise ValueError("Bailian API key not found in config")
            
        self.bailian_api_key = self.config["api_keys"]["bailian_api_key"]
        self.qwen_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _get_vertex_token(self):
        if 'service_account_path' in self.config['vertex']:
            credentials = service_account.Credentials.from_service_account_file(
                self.config['vertex']['service_account_path'])
            return credentials.token
        import subprocess
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"]).decode().strip()

    def test_model(self, task: str, style: str, case: Dict, shot_mode: str = 'zero', num_shots: int = 0) -> Dict:
        try:
            template = self.prompt_manager.load_prompt(task, style, shot_mode, num_shots)
            formatted_prompt = self.prompt_manager.format_prompt(template, case)
            return self._get_model_response(formatted_prompt)
        except Exception as e:
            logging.error(f"Test error: {str(e)}")
            return None

    def _get_model_response(self, prompt: str) -> Dict:
        try:
            if "gemini" in self.model_name:
                return self._test_gemini(prompt)
            elif "claude" in self.model_name:
                return self._test_claude(prompt)
            elif "gpt" in self.model_name:
                return self._test_openai(prompt)
            elif "meta-llama" in self.model_name:
                return self._test_vertex_llama(prompt)
            elif "grok" in self.model_name:
                return self._test_grok(prompt)
            elif "glm" in self.model_name:
                return self._test_glm(prompt)
            elif "deepseek" in self.model_name:
                return self._test_deepseek(prompt)
            elif "qwen" in self.model_name:
                return self._test_qwen(prompt)
            elif "bailian-llama" in self.model_name:
                return self._test_bailian(prompt)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        except Exception as e:
            logging.error(f"Response error: {str(e)}")
            return None

    def _test_gemini(self, prompt: str) -> Dict:
        """Test Gemini model with rate limiting"""
        try:
            # Enforce rate limit before making request
            self._enforce_rate_limit()
            
            response = self.chat.send_message(prompt)
            return self._parse_json_response(response.text)
            
        except Exception as e:
            if "429" in str(e):  # Quota exceeded error
                logging.warning("Rate limit exceeded, implementing longer delay...")
                time.sleep(60)  # Wait a full minute before retry
                try:
                    self._enforce_rate_limit()
                    response = self.chat.send_message(prompt)
                    return self._parse_json_response(response.text)
                except Exception as retry_error:
                    logging.error(f"Retry failed: {str(retry_error)}")
                    return None
            else:
                logging.error(f"Gemini error: {str(e)}")
                return None

    def process_gemini_batch(self, prompts: List[str], batch_size: int = 1) -> List[Dict]:
        """Process a batch of prompts with strict rate limiting"""
        results = []
        for prompt in prompts:
            result = self._test_gemini(prompt)
            if result:
                results.append(result)
                
        return results
        
    def _test_grok(self, prompt: str) -> Dict:
        """Test Grok model through X.AI API"""
        try:
            url = "https://api.x.ai/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.xai_api_key}"
            }
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "grok-beta",
                "stream": False,
                "temperature": 0.3  
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                return self._parse_json_response(response_data['choices'][0]['message']['content'])
            else:
                logging.error(f"Grok API request failed: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Grok API error: {str(e)}")
            return None

    def test_model(self, task: str, style: str, case: Dict, shot_mode: str = 'zero', num_shots: int = 0) -> Dict:
        try:
            template = self.prompt_manager.load_prompt(task, style, shot_mode, num_shots)
            formatted_prompt = self.prompt_manager.format_prompt(template, case)
            return self._get_model_response(formatted_prompt)
        except Exception as e:
            logging.error(f"Test error: {str(e)}")
            return None

    def _test_claude(self, prompt: str) -> Dict:
        """Test Claude model through direct API or Vertex AI"""
        try:
            if self.config.get("use_vertex_api", False):
                return self._test_vertex_claude(prompt)
            else:
                # Get correct model identifier for direct API
                model_id = self.MODEL_MAPPINGS.get(self.model_name)
                if not model_id:
                    raise ValueError(f"Unsupported model: {self.model_name}")

                response = self.client.messages.create(
                    model=model_id,  # Use mapped model identifier
                    max_tokens=2048,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.3
                )
                return self._parse_json_response(response.content[0].text)
        except Exception as e:
            logging.error(f"Claude error: {str(e)}")
            return None

    def _test_vertex_claude(self, prompt: str) -> Dict:
        """Test Claude model through Vertex AI"""
        try:
            vertex_model_name = f"{self.model_name}-vertex"
            model_id = self.MODEL_MAPPINGS.get(vertex_model_name)
            if not model_id:
                raise ValueError(f"Unsupported Vertex AI model: {vertex_model_name}")

            url = f"{self.endpoint}/v1/projects/{self.config['vertex']['project_id']}/locations/{self.config['vertex']['location']}/publishers/anthropic/models/{model_id}:rawPredict"
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "max_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 1,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return self._parse_json_response(response.json()['outputs'][0])
            else:
                logging.error(f"Request failed: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Vertex Claude error: {str(e)}")
            return None

    def _test_openai(self, prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return self._parse_json_response(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"OpenAI error: {str(e)}")
            return None

    def _test_vertex_llama(self, prompt: str) -> Dict:
        """Test Llama model through Vertex AI"""
        try:
            model_id = self.MODEL_MAPPINGS.get(self.model_name)
            if not model_id:
                raise ValueError(f"Unsupported model: {self.model_name}")

            url = f"https://{self.config['vertex']['location']}-aiplatform.googleapis.com/v1/projects/{self.config['vertex']['project_id']}/locations/{self.config['vertex']['location']}/endpoints/openapi/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_id,
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 2048,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return self._parse_json_response(response.json().get('choices', [{}])[0].get('message', {}).get('content', ''))
            else:
                logging.error(f"Request failed: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Llama API error: {str(e)}")
            return None

    def _test_deepseek(self, prompt: str) -> Dict:
        """Test DeepSeek model through API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.MODEL_MAPPINGS.get(self.model_name, self.model_name),
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.3,
                "max_tokens": 2048,
                "stream": False
            }
            
            # Add retry mechanism with exponential backoff
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.deepseek_api_url, 
                        headers=headers, 
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        return self._parse_json_response(
                            response_data['choices'][0]['message']['content']
                        )
                    elif response.status_code == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        logging.warning(
                            f"Rate limit reached. Waiting {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logging.error(
                            f"DeepSeek API request failed: {response.status_code}"
                        )
                        logging.error(f"Response: {response.text}")
                        return None
                        
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        logging.error("DeepSeek API request timed out after all retries")
                        return None
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
        except Exception as e:
            logging.error(f"DeepSeek API error: {str(e)}")
            return None

    def _test_glm(self, prompt: str) -> Dict:
        """Test GLM model with comprehensive error handling and retries"""
        try:
            headers = {
                "Authorization": f"Bearer {self.glm_api_key}",
                "Content-Type": "application/json"
            }
            
            model_config = self.glm_configs.get(self.model_name, {
                "max_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.7
            })
            
            data = {
                "model": self.MODEL_MAPPINGS.get(self.model_name, self.model_name),
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                **model_config  
            }
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.glm_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        content = response_data['choices'][0]['message']['content']
                        return self._parse_json_response(content)
                        
                    elif response.status_code == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        logging.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                        
                    elif response.status_code == 413:  # Content too large
                        logging.error("Content too large for model")
                        return {"error": "Content too large for model"}
                        
                    else:
                        logging.error(f"GLM API request failed: {response.status_code}")
                        logging.error(f"Response: {response.text}")
                        return None
                        
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        logging.error("GLM API request timed out after all retries")
                        return None
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request failed: {str(e)}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(retry_delay * (2 ** attempt))
                    
        except Exception as e:
            logging.error(f"GLM API error: {str(e)}")
            return None

    def process_glm_batch(self, prompts: List[str], batch_size: int = 5) -> List[Dict]:
        """Process a batch of prompts with rate limiting"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                result = self._test_glm(prompt)
                if result:
                    batch_results.append(result)
                time.sleep(1)  # Rate limiting between requests
            
            results.extend(batch_results)
            
            # Add delay between batches
            if i + batch_size < len(prompts):
                time.sleep(5)  # Delay between batches
                
        return results

    def _test_bailian(self, prompt: str) -> Dict:
        """Test Llama model through Aliyun Bailian API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.bailian_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.MODEL_MAPPINGS.get(self.model_name, self.model_name),
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.bailian_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        logging.debug(f"Raw response: {response_data}")  # Debug log
                        
                        # Handle Bailian's response format
                        if 'output' in response_data:
                            if 'choices' in response_data['output']:
                                content = response_data['output']['choices'][0]['message']['content']
                            elif 'message' in response_data['output']:
                                content = response_data['output']['message']['content']
                            elif 'text' in response_data['output']:
                                content = response_data['output']['text']
                            else:
                                logging.error(f"Unexpected response structure: {response_data}")
                                return None
                            
                            return self._parse_json_response(content)
                            
                        logging.error(f"Missing 'output' in response: {response_data}")
                        return None
                        
                    elif response.status_code == 429:
                        wait_time = retry_delay * (2 ** attempt)
                        logging.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                        
                    else:
                        logging.error(f"Bailian API request failed: {response.status_code}")
                        logging.error(f"Response: {response.text}")
                        return None
                        
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        logging.error("Bailian API request timed out after all retries")
                        return None
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
        except Exception as e:
            logging.error(f"Bailian API error: {str(e)}")
            return None

    def _test_qwen(self, prompt: str) -> Dict:
        """Test Qwen model through Aliyun Bailian API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.bailian_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.MODEL_MAPPINGS.get(self.model_name, self.model_name),
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.qwen_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        logging.debug(f"Raw response: {response_data}") 
                        
                        if 'output' in response_data:
                            if 'choices' in response_data['output']:
                                content = response_data['output']['choices'][0]['message']['content']
                            elif 'message' in response_data['output']:
                                content = response_data['output']['message']['content']
                            elif 'text' in response_data['output']:
                                content = response_data['output']['text']
                            else:
                                logging.error(f"Unexpected response structure: {response_data}")
                                return None
                            
                            return self._parse_json_response(content)
                            
                        logging.error(f"Missing 'output' in response: {response_data}")
                        return None
                        
                    elif response.status_code == 429:
                        wait_time = retry_delay * (2 ** attempt)
                        logging.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                        
                    else:
                        logging.error(f"Qwen API request failed: {response.status_code}")
                        logging.error(f"Response: {response.text}")
                        return None
                        
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        logging.error("Qwen API request timed out after all retries")
                        return None
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
        except Exception as e:
            logging.error(f"Qwen API error: {str(e)}")
            return None

    def _parse_json_response(self, response_text: str) -> Dict:
        try:
            if '```' in response_text:
                response_text = response_text.replace('```json', '').replace('```', '')
            
            start = response_text.find('{')
            end = response_text.rfind('}')
            
            if start != -1 and end != -1:
                response_text = response_text[start:end + 1].strip()
                return json.loads(response_text)
            else:
                logging.error("No JSON found in response")
                return {"error": "No JSON content found"}
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error: {str(e)}")
            return {"error": f"Failed to parse JSON: {str(e)}"}
