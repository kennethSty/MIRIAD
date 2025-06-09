from typing import Any
import json
import os
import random
from openai import OpenAI

from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import logging

from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import re
from dataclasses import dataclass
import asyncio

def locate_problematic_data_start(text):
    # find the start of the extra data
    match = re.search(r"\(char (\d+)\)", text)
    if match:
        number = int(match.group(1))
        return number
    else:
        # if the extra data start is not found, return -1
        return -1
    
def search_for_choice(input_text):
    # Regex to extract the value of "choice"
    match = re.search(r'"choice":\s*"(\w)"', input_text)

    if match:
        choice_value = match.group(1)
        return choice_value
    else:
        return None

def llm_output_str2json(llm_output: str, max_retry: int = 1):
    error_message = None
    num_retry = 0
    # Step 1: Escape newlines and clean whitespace
    cleaned_output = re.sub(r'\n\s+', ' ', llm_output)  # Replace newlines followed by spaces with a single space
    cleaned_output = re.sub(r'\n', ' ', cleaned_output)  # Escape remaining newlines

    # Step 2: Ensure keys and values are properly quoted
    cleaned_output = re.sub(r'(["\'])(?=\w+\s*:)', r'"', cleaned_output)  # Ensure keys use double quotes
    cleaned_output = re.sub(r"(?<!\w)'(\w+)':", r'"\1":', cleaned_output) # Replace single quotes around keys with double quotes
    cleaned_output = re.sub(r'(?<=: )\'(.*?)\'', r'"\1"', cleaned_output)  # Ensure values use double quotes

    # Step 3: Remove trailing commas
    cleaned_output = re.sub(r',\s*}', '}', cleaned_output)  # Remove trailing commas in objects
    cleaned_output = re.sub(r',\s*]', ']', cleaned_output)  # Remove trailing commas in arrays
    
    # Step 3.1: Replace any backtick after colon+space or before a closing brace
    cleaned_output = re.sub(r'(:\s*)`|`(?=\s*})', r'\1"', cleaned_output)

    # Step 4: Parse and validate the cleaned JSON string
    while num_retry <= max_retry:
        try:
            parsed_json = json.loads(cleaned_output)
            return parsed_json, error_message
        except json.JSONDecodeError as e:
            if 'Extra data' in str(e): # model eloborates beyond the json string. Remove the extra data
                    # find the start of the extra data
                    extra_data_start_idx = locate_problematic_data_start(str(e)) # extract the start index of extra data from error message
                    cleaned_output = cleaned_output[:extra_data_start_idx]
            try:
                answer_value = cleaned_output.split("\"answer\": \"")[-1].split("\", \"choice\":")[0]
                # replace all the double quotes with single quotes in answer_value
                new_answer_value = answer_value.replace('\"', "\'")
                # replace the old answer_value with the new one in t
                cleaned_output = cleaned_output.replace(answer_value, new_answer_value)
                parsed_json = json.loads(cleaned_output)
                return parsed_json, error_message
            except Exception as e:
                error_message = {"str2json_error": str(e), "cleaned_output": cleaned_output, "llm_output": llm_output}
                print(f"Error decoding JSON from LLM output: {e}")
                num_retry += 1
    choice = search_for_choice(cleaned_output) 
    if num_retry != 0 and choice:
        print(f"Failed to parse JSON after {num_retry} retries. Extracted choice value: {choice}")
        print(f"Original cleaned_output: {cleaned_output}")
    if choice:
        return {"choice": choice, "answer": "Not Available due to parsing failure"}, None # return the choice value and a default answer, then pass the error message as None
    else:
        return None, error_message
    
def extract_binary_class(response_text):
    error_message = None
    matches = re.findall(r'\b[01]\b', response_text)
    if matches and len(matches) == 1:
        parsed = matches[0]
    elif matches and len(matches) > 1:
        error_message = 'Chatty model returned more than 1 binary classification (0 or 1)'
        parsed = '-2' # more than 1 binary classification
    else:
        error_message = "No valid binary classification (0 or 1) found in response."
        parsed = "-1" # no valid binary classification
    return {"choice": parsed, "response": response_text}, error_message
    
class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def generate(self, prompts: List[Dict[str, str]], **kwargs) -> List[Dict]:
        """Generate responses for a batch of prompts."""
        pass
    
@dataclass
class LLMRequest:
    """Unified format for LLM requests."""
    system: str
    user: str
    request_id: str
    
    # Optional model-specific parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

@dataclass
class LLMResponse:
    """Unified format for LLM responses."""
    request_id: str
    content: Any  # Can handle both string and List[ContentBlock]
    parsed_response: Optional[Dict] = None
    error: Optional[str] = None

class ClaudeLLM(BaseLLM):
    """Claude-specific LLM implementation with batch processing capabilities."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_concurrent_requests: int = 50,
        rate_limit_per_minute: int = 500,
        batch_size: int = 100
    ):
        self.model = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_period = 60 / rate_limit_per_minute
        self.batch_size = batch_size
        self.last_request_time = 0
        
    def generate(self, prompts: List[Dict[str, str]], **kwargs) -> List[Dict]:
        """
        Generate responses for a batch of prompts.
        Implements the BaseLLM interface method.
        """
        # Convert prompts to LLMRequests
        requests = [
            LLMRequest(
                system=prompt["system"],
                user=prompt["user"],
                request_id=str(i)
            )
            for i, prompt in enumerate(prompts)
        ]
        
        # Run batch inference
        responses = self._run_batch_inference(requests)
        
        # Convert responses to the expected format
        return [
            self.parse_response(resp) for resp in responses
        ]
    
    def parse_response(self, response: LLMResponse) -> Dict:
        """Parse Claude response into standardized format."""
        if response.error:
            return {"error": response.error, "choice": "Z"}
        
        try:
            parsed, str2json_error = llm_output_str2json(response.content)
            if str2json_error:
                return {"error": str2json_error, "choice": "Z"}
            if not isinstance(parsed, dict) or "choice" not in parsed.keys():
                return {"error": f"Invalid response format {type(parsed)}", "choice": "Z"}
            response.parsed_response = parsed
            return parsed # return the parsed response when there is no error
        except Exception as e:
            return {"error": str(e), "choice": "Z"}

    def _run_batch_inference(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Run batch inference using async processing."""
        return asyncio.run(self._process_all_requests(requests))
    
    async def _process_all_requests(
        self,
        requests: List[LLMRequest]
    ) -> List[LLMResponse]:
        """Process all requests with batching and concurrent execution."""
        # Initialize async client
        client = AsyncAnthropic()
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_single_request(request: LLMRequest, max_retries: int = 3) -> LLMResponse:
            retry_count = 0
            base_delay = 1
            while retry_count < max_retries:
                try:
                    async with semaphore:
                        # Rate limiting
                        current_time = time.time()
                        time_since_last = current_time - self.last_request_time
                        if time_since_last < self.rate_limit_period:
                            await asyncio.sleep(self.rate_limit_period - time_since_last)
                        self.last_request_time = time.time()
                        
                        # Make API call
                        response = await client.messages.create(
                            model=self.model,
                            max_tokens=1000,
                            temperature=0,
                            system=request.system,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": request.user,
                                        }
                                    ]
                                }
                            ]
                        )
                        
                        return LLMResponse(
                            request_id=request.request_id,
                            content=response.content[0].text
                        )
                        
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str and retry_count < max_retries: # retry if hit rate limit error
                        # Retry with exponential backoff
                        retry_count += 1
                        delay = base_delay * (2 ** (retry_count - 1))
                        # Add jitter to prevent thundering herd
                        jitter = random.uniform(0, 0.1 * delay)
                        await asyncio.sleep(delay + jitter)
                    return LLMResponse(
                        request_id=request.request_id,
                        content=[],
                        error=str(e)
                    )
            return LLMResponse(
                request_id=request.request_id,
                content=[],
                error="Max retries exceeded"
    )
        
        # Split requests into batches
        batches = [
            requests[i:i + self.batch_size] 
            for i in range(0, len(requests), self.batch_size)
        ]
        
        # Process batches with progress bar
        all_responses = []
        for batch in tqdm_asyncio(batches, desc="Processing batches"):
            batch_responses = await asyncio.gather(
                *[process_single_request(req) for req in batch]
            )
            all_responses.extend(batch_responses)
        
        return all_responses

class HuggingFaceLLM(BaseLLM):
    """Local HuggingFace LLM implementation with optimized multi-GPU inference."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 32768,
        rank: int = 0,
        world_size: int = 8,
        device_id=None,
        **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id if device_id is not None else rank  # Store device_id
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model and tokenizer with optimal configurations for DDP."""
        self.logger.info(f"Initializing model {self.model_name} on GPU {self.device_id}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimized settings
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        ).to(self.device_id)  # Use device_id instead of rank
        
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Model initialized on GPU {self.device_id}")
    
    def _prepare_messages(self, system: str, user: str) -> List[Dict[str, str]]:
        """Prepare messages in the chat format."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages
    
    def _batch_generate(self, batch_messages: List[List[Dict]], batch_size: int, rank: int, world_size: int) -> List[str]:
        """Generate responses for a batch of messages with DDP."""
        all_outputs = []
        
        for i in tqdm(range(0, len(batch_messages), batch_size), desc=f"GPU {rank} Generating responses"):
            current_batch = batch_messages[i:i + batch_size]
            # Convert messages to chat template format
            chat_texts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False)
                for messages in current_batch
            ]
            
            # Tokenize batch with simple padding
            tokenized = self.tokenizer(
                chat_texts,
                truncation=True,
                padding="longest",  # Only pad to longest in batch
                max_length=self.max_length,
                return_tensors="pt"
            ).to(rank)
            
            # Move tensors to device (already moved since above)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            
            # Generate with optimized settings
            with torch.no_grad():  # No need for autocast since we're already using bfloat16
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=(self.temperature > 0),
                )
            
            # Batch decode the outputs
            input_lengths = [input_ids[j].shape[0] for j in range(len(outputs))]
            new_tokens = [outputs[j][input_lengths[j]:] for j in range(len(outputs))]
            
            decoded_outputs = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            all_outputs.extend(decoded_outputs)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_outputs
    
    def generate(self, prompts: List[Dict[str, str]], rank: int, world_size: int, **kwargs) -> List[Dict]:
        """Generate responses for a batch of prompts with DDP."""
        self.logger.info(f"GPU {rank}: Generating responses for {len(prompts)} prompts")
        
        batch_messages = [
            self._prepare_messages(prompt["system"], prompt["user"])
            for prompt in prompts
        ]
        
        
        responses = self._batch_generate(batch_messages, self.batch_size, self.rank, self.world_size)
        
        return [
            self.parse_response(
                LLMResponse(
                    request_id=str(i),
                    content=response
                )
            )
            for i, response in enumerate(responses)
        ]
    
    def parse_response(self, response: LLMResponse) -> Dict:
        """Parse model response into standardized format."""
        if response.error:
            return {"error": response.error, "choice": "Z"}
        
        try:
            parsed, str2json_error = llm_output_str2json(response.content)
            if str2json_error:
                return {"error": str2json_error, "choice": "Z"}
            if not isinstance(parsed, dict) or "choice" not in parsed.keys():
                return {"error": f"Invalid response format {type(parsed)}", "choice": "Z"}
            response.parsed_response = parsed
            return parsed # return the parsed response when there is no error
        except Exception as e:
            return {"error": str(e), "choice": "Z"}
        
class MistralLLM(HuggingFaceLLM):
    """Local Mistral LLM implementation with optimized multi-GPU inference."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 32768,
        rank: int = 0,
        world_size: int = 8,
        device_id=None,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            device_id=device_id if device_id is not None else rank,  # Store device_id
            **kwargs
        )
        
class Llama3LLM(HuggingFaceLLM):
    """Local Llama LLM implementation inheriting from MistralLLM."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",  # Default to Llama 3.1 8B Instruct
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 8192,  # Default to original training length
        rank: int = 0,
        world_size: int = 8,
        device_id=None,
        **kwargs
    ):
         super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            device_id=device_id if device_id is not None else rank,  # Store device_id
            **kwargs
        )
    
    def _initialize_model(self):
        """Initialize model and tokenizer with optimal configurations for DDP."""
         # Add any Llama3-specific initialization if needed
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing model {self.model_name} on GPU {self.device_id}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        
        # Load model with optimized settings
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        ).to(self.device_id)
        
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Model initialized on GPU {self.device_id}")

    def _batch_generate(self, batch_messages: List[List[Dict]], batch_size: int, rank: int, world_size: int) -> List[str]:
        """Generate responses for a batch of messages with DDP."""
        all_outputs = []
        
        for i in tqdm(range(0, len(batch_messages), batch_size), desc=f"GPU {rank} Generating responses"):
            current_batch = batch_messages[i:i + batch_size]
            # Convert messages to chat template format
            chat_texts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False)
                for messages in current_batch
            ]
            
            # Tokenize batch with simple padding
            tokenized = self.tokenizer(
                chat_texts,
                truncation=True,
                padding="longest",  # Only pad to longest in batch
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device_id)
            
            # Move tensors to device (already moved since above)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            
            # Generate with optimized settings
            with torch.no_grad():  # No need for autocast since we're already using bfloat16
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=(self.temperature > 0),
                )
                            
            # Batch decode the outputs
            input_lengths = [input_ids[j].shape[0] for j in range(len(outputs))]
            new_tokens = [outputs[j][input_lengths[j]+3:] for j in range(len(outputs))] # remove the first 3 tokens in assistant answer [28006,  78191, 128007] for <|start_header_id|>assistant<|end_header_id|>
            
            decoded_outputs = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            all_outputs.extend(decoded_outputs)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_outputs
    
# ---- for MedHallu Benchmark, binary classification ----
class HuggingFaceLLM_BinaryClassification(HuggingFaceLLM):
    """Local HuggingFace LLM implementation with optimized multi-GPU inference for binary classification."""

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 32768,
        rank: int = 0,
        world_size: int = 8,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            **kwargs
        )
    def parse_response(self, response: LLMResponse) -> Dict:
        """Parse LLM response into standardized format, 
            which should be 0 or 1 for binary classification. -1 indicates error."""
        print()
        if response.error:
            return {"error": response.error, "choice": "-1"}
        
        try:
            text_content = response.content
            parsed, extraction_error = extract_binary_class(text_content)
            response.parsed_response = parsed
            if not extraction_error:
                return parsed
            else:
                return {"error": extraction_error, "choice": parsed["choice"]}
        except Exception as e:
            return {"error": str(e), "choice": "-1"}

class MistralLLM_BinaryClassification(HuggingFaceLLM_BinaryClassification):
    """Local Mistral LLM implementation with optimized multi-GPU inference."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 32768,
        rank: int = 0,
        world_size: int = 8,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            **kwargs
        )

class Llama3LLM_BinaryClassification(HuggingFaceLLM_BinaryClassification):
    """Local Llama LLM implementation inheriting from MistralLLM."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",  # Default to Llama 3.1 8B Instruct
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_length: int = 8192,  # Default to original training length
        rank: int = 0,
        world_size: int = 8,
        **kwargs
    ):
         super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            **kwargs
        )
    
    def _initialize_model(self):
        """Initialize model and tokenizer with optimal configurations for DDP."""
         # Add any Llama3-specific initialization if needed
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing model {self.model_name} on GPU {self.rank}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        
        # Load model with optimized settings
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        ).to(self.rank)
        
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Model initialized on GPU {self.rank}")

    def _batch_generate(self, batch_messages: List[List[Dict]], batch_size: int, rank: int, world_size: int) -> List[str]:
        """Generate responses for a batch of messages with DDP."""
        all_outputs = []
        
        for i in tqdm(range(0, len(batch_messages), batch_size), desc=f"GPU {rank} Generating responses"):
            current_batch = batch_messages[i:i + batch_size]
            # Convert messages to chat template format
            chat_texts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False)
                for messages in current_batch
            ]
            
            # Tokenize batch with simple padding
            tokenized = self.tokenizer(
                chat_texts,
                truncation=True,
                padding="longest",  # Only pad to longest in batch
                max_length=self.max_length,
                return_tensors="pt"
            ).to(rank)
            
            # Move tensors to device (already moved since above)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            
            # Generate with optimized settings
            with torch.no_grad():  # No need for autocast since we're already using bfloat16
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=(self.temperature > 0),
                )
                            
            # Batch decode the outputs
            input_lengths = [input_ids[j].shape[0] for j in range(len(outputs))]
            new_tokens = [outputs[j][input_lengths[j]+3:] for j in range(len(outputs))] # remove the first 3 tokens in assistant answer [28006,  78191, 128007] for <|start_header_id|>assistant<|end_header_id|>
            
            decoded_outputs = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            all_outputs.extend(decoded_outputs)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_outputs