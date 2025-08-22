#!/usr/bin/env python3
"""
Multi-Model Job Search Evaluation Script
Supports loading and comparing different model types for job search and retrieval tasks.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import time


class ModelLoader:
    """Handles loading different types of models for job search evaluation."""
    
    SUPPORTED_TYPES = [
        "sentence-transformer",
        "sentence-transformer-recovered", 
        "huggingface-auto",
        "distilbert"
    ]
    
    def __init__(self, device: str = None):
        """Initialize ModelLoader with device selection."""
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")
    
    def load_model(self, model_path: str, model_type: str) -> Any:
        """
        Load a model based on the specified type.
        
        Args:
            model_path: Path to the model directory or file
            model_type: Type of model to load
            
        Returns:
            Loaded model object
        """
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {self.SUPPORTED_TYPES}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"Loading {model_type} model from: {model_path}")
        
        try:
            if model_type == "sentence-transformer":
                return self._load_sentence_transformer(model_path)
            elif model_type == "sentence-transformer-recovered":
                return self._load_sentence_transformer_recovered(model_path)
            elif model_type == "huggingface-auto":
                return self._load_huggingface_auto(model_path)
            elif model_type == "distilbert":
                return self._load_distilbert(model_path)
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            raise
    
    def _load_sentence_transformer(self, model_path: Path) -> SentenceTransformer:
        """Load standard SentenceTransformer model."""
        model = SentenceTransformer(str(model_path), trust_remote_code=True)
        print(f"✓ SentenceTransformer model loaded successfully")
        return model
    
    def _load_sentence_transformer_recovered(self, model_path: Path) -> SentenceTransformer:
        """Load SentenceTransformer model with recovered weights (.pt file)."""
        # Load base SentenceTransformer
        model = SentenceTransformer(str(model_path), trust_remote_code=True)
        
        # Look for recovered weights
        recovered_files = list(model_path.glob("*.pt"))
        if not recovered_files:
            print("Warning: No .pt files found, using base model only")
            return model
        
        # Use the first .pt file found (or look for specific naming pattern)
        recovered_path = recovered_files[0]
        for pt_file in recovered_files:
            if "recovered" in pt_file.name.lower():
                recovered_path = pt_file
                break
        
        print(f"Loading recovered weights from: {recovered_path}")
        try:
            state_dict = torch.load(recovered_path, map_location="cpu")
            model._first_module().auto_model.load_state_dict(state_dict, strict=False)
            print("✓ SentenceTransformer model with recovered weights loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load recovered weights: {e}")
            print("Using base model without recovered weights")
        
        return model
    
    def _load_huggingface_auto(self, model_path: Path) -> AutoModel:
        """Load model using HuggingFace AutoModel."""
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        model.to(self.device)
        print(f"✓ HuggingFace AutoModel loaded successfully")
        return model
    
    def _load_distilbert(self, model_path: Path) -> Any:
        """Load DistilBERT model for token classification."""
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model.to(self.device)
        model.eval()
        print(f"✓ DistilBERT model loaded successfully")
        return {"model": model, "tokenizer": tokenizer}


class JobSearchEngine:
    """Handles job search and retrieval using loaded models."""
    
    def __init__(self, model: Any, model_type: str, device: str = None, cache_dir: str = "embedding_cache"):
        """Initialize JobSearchEngine with a loaded model."""
        self.model = model
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.jobs = []
        self.documents = []
        self.job_embeddings = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_jobs(self, jobs_path: str, max_desc_chars: int = 1000):
        """Load and preprocess job data from JSON file."""
        jobs_path = Path(jobs_path)
        if not jobs_path.exists():
            raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
        
        print(f"Loading jobs from: {jobs_path}")
        with open(jobs_path, 'r', encoding='utf-8') as f:
            self.jobs = json.load(f)
        
        # Preprocess jobs into search documents
        self.documents = []
        for job in self.jobs:
            title = job.get('job_title', '')
            company = job.get('company_name', '')
            location = job.get('location_raw', '')
            remote = job.get('location_type', '') == 'Remote'
            salary = job.get('salary_range', '') or job.get('salary', '')
            job_type = job.get('job_type', '')
            description = self._clean_description(job.get('description', ''))[:max_desc_chars]
            
            text = f"""search_document:
Title: {title}
Company: {company}
Location: {location}
Remote: {remote}
Job Type: {job_type}
Salary: {salary}
Description: {description}"""
            
            self.documents.append(text)
        
        print(f"✓ Loaded {len(self.documents)} job documents")
    
    def _clean_description(self, html_text: str) -> str:
        """Clean HTML from job descriptions."""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    
    def _generate_cache_key(self, jobs_path: str, model_identifier: str) -> str:
        """Generate a unique cache key based on jobs file and model."""
        # Create hash from jobs file path, modification time, and model identifier
        jobs_path = Path(jobs_path)
        if jobs_path.exists():
            mtime = str(jobs_path.stat().st_mtime)
        else:
            mtime = "0"
        
        # Include documents content in hash for extra safety
        documents_hash = hashlib.md5(''.join(self.documents[:10]).encode()).hexdigest()[:8]
        
        key_string = f"{jobs_path.name}_{mtime}_{model_identifier}_{len(self.documents)}_{documents_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_model_identifier(self) -> str:
        """Get a unique identifier for the current model."""
        if hasattr(self.model, '_modules'):
            # For SentenceTransformer models
            model_info = str(self.model._modules)[:100]
        elif hasattr(self.model, 'config'):
            # For HuggingFace models
            model_info = str(self.model.config)[:100]
        else:
            model_info = str(type(self.model))
        
        return f"{self.model_type}_{hashlib.md5(model_info.encode()).hexdigest()[:8]}"
    
    def _save_embeddings_cache(self, cache_key: str):
        """Save embeddings to cache."""
        if self.job_embeddings is None or self.cache_dir.name == "no_cache":
            return
        
        cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
        cache_data = {
            'embeddings': self.job_embeddings.cpu() if isinstance(self.job_embeddings, torch.Tensor) else self.job_embeddings,
            'documents': self.documents,
            'model_type': self.model_type,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Embeddings cached to: {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache embeddings: {e}")
    
    def _load_embeddings_cache(self, cache_key: str) -> bool:
        """Load embeddings from cache. Returns True if successful."""
        if self.cache_dir.name == "no_cache":
            return False
            
        cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data
            if (cache_data['documents'] != self.documents or 
                cache_data['model_type'] != self.model_type):
                print("⚠️ Cache invalidated: documents or model type mismatch")
                return False
            
            # Load embeddings
            embeddings = cache_data['embeddings']
            if isinstance(embeddings, torch.Tensor):
                self.job_embeddings = embeddings.to(self.device)
            else:
                self.job_embeddings = torch.tensor(embeddings).to(self.device)
            
            cache_time = cache_data.get('timestamp', 'unknown')
            print(f"✓ Loaded cached embeddings from: {cache_file} (cached: {cache_time})")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load cached embeddings: {e}")
            return False
    
    def encode_jobs(self, batch_size: int = 64, jobs_path: str = None, force_recompute: bool = False):
        """Encode all job documents into embeddings with caching support."""
        if not self.documents:
            raise ValueError("No job documents loaded. Call load_jobs() first.")
        
        # Generate cache key
        model_identifier = self._get_model_identifier()
        cache_key = self._generate_cache_key(jobs_path or "unknown", model_identifier)
        
        # Try to load from cache first (unless force_recompute is True)
        if not force_recompute and self._load_embeddings_cache(cache_key):
            print("✓ Using cached embeddings (skipped encoding)")
            return
        
        print("Encoding job embeddings...")
        start_time = time.time()
        
        if self.model_type in ["sentence-transformer", "sentence-transformer-recovered"]:
            self.job_embeddings = self.model.encode(
                self.documents,
                convert_to_tensor=True,
                device=self.device,
                batch_size=batch_size,
                show_progress_bar=True
            )
            # Apply normalization as in original search_simulator.py
            self.job_embeddings = F.layer_norm(self.job_embeddings, (self.job_embeddings.shape[-1],))
            self.job_embeddings = F.normalize(self.job_embeddings, p=2, dim=1)
            
        elif self.model_type == "huggingface-auto":
            # For HuggingFace AutoModel, we need to implement custom encoding
            raise NotImplementedError("HuggingFace AutoModel encoding not yet implemented")
            
        elif self.model_type == "distilbert":
            # DistilBERT is primarily for token classification, not sentence embeddings
            raise NotImplementedError("DistilBERT encoding for job search not yet implemented")
        
        encoding_time = time.time() - start_time
        print(f"✓ Job embeddings encoded successfully in {encoding_time:.2f} seconds")
        
        # Save to cache
        self._save_embeddings_cache(cache_key)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for jobs using the given query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with job information and scores
        """
        if self.job_embeddings is None:
            raise ValueError("Job embeddings not computed. Call encode_jobs() first.")
        
        print(f"Searching for: '{query}'")
        
        # Encode query
        query_text = f"search_query: {query}"
        
        if self.model_type in ["sentence-transformer", "sentence-transformer-recovered"]:
            query_embedding = self.model.encode(
                query_text,
                convert_to_tensor=True,
                device=self.device
            )
            # Apply same normalization as jobs
            query_embedding = F.layer_norm(query_embedding, (query_embedding.shape[0],))
            query_embedding = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            
            # Compute similarities
            similarities = F.cosine_similarity(query_embedding, self.job_embeddings)
            top_indices = np.argsort(-similarities.cpu().numpy())
            
        else:
            raise NotImplementedError(f"Search not implemented for model type: {self.model_type}")
        
        # Prepare results
        results = []
        for i in range(min(top_k, len(top_indices))):
            idx = top_indices[i]
            score = float(similarities[idx])
            job = self.jobs[idx]
            
            result = {
                "rank": i + 1,
                "score": score,
                "job_id": int(idx),
                "job_title": job.get('job_title', ''),
                "company_name": job.get('company_name', ''),
                "location_raw": job.get('location_raw', ''),
                "job_type": job.get('job_type', ''),
                "salary_range": job.get('salary_range', '') or job.get('salary', ''),
                "description": job.get('description', '')
            }
            results.append(result)
        
        return results


class ModelEvaluator:
    """Main evaluation orchestrator for comparing multiple models."""
    
    def __init__(self, args):
        """Initialize evaluator with command line arguments."""
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize model loader
        self.model_loader = ModelLoader()
        self.search_engines = {}
    
    def load_models(self):
        """Load all specified models."""
        models_to_load = []
        
        # Collect model specifications
        if self.args.model_1 and self.args.model_1_type:
            models_to_load.append(("model1", self.args.model_1, self.args.model_1_type))
        
        if self.args.model_2 and self.args.model_2_type:
            models_to_load.append(("model2", self.args.model_2, self.args.model_2_type))
        
        if not models_to_load:
            raise ValueError("At least one model must be specified")
        
        # Load models and create search engines
        for model_name, model_path, model_type in models_to_load:
            print(f"\n{'='*50}")
            print(f"Loading {model_name}: {model_path} ({model_type})")
            print(f"{'='*50}")
            
            try:
                model = self.model_loader.load_model(model_path, model_type)
                if self.args.no_cache:
                    search_engine = JobSearchEngine(model, model_type, cache_dir="no_cache")
                else:
                    search_engine = JobSearchEngine(model, model_type, cache_dir=self.args.cache_dir)
                search_engine.load_jobs(self.args.jobs_data_path)
                
                if self.args.no_cache:
                    # Skip caching entirely
                    search_engine.encode_jobs()
                else:
                    # Use caching system
                    search_engine.encode_jobs(
                        jobs_path=self.args.jobs_data_path,
                        force_recompute=self.args.force_recompute
                    )
                
                self.search_engines[model_name] = search_engine
                print(f"✓ {model_name} ready for evaluation")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                raise
    
    def run_search_evaluation(self, query: str = "Python Developer in San Francisco with Salary 100000"):
        """Run search evaluation on all loaded models."""
        print(f"\n{'='*60}")
        print(f"RUNNING SEARCH EVALUATION")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        for model_name, search_engine in self.search_engines.items():
            print(f"\n{'-'*40}")
            print(f"Evaluating {model_name}")
            print(f"{'-'*40}")
            
            # Run search
            results = search_engine.search(query, top_k=10)
            
            # Display top 5 results
            print("\nTop 5 results:")
            for i, result in enumerate(results[:5]):
                print(f"  {result['rank']}. ({result['score']:.3f}) {result['job_title']} at {result['company_name']} | {result['location_raw']}")
            
            # Export results
            export_data = {"query": query, "results": results}
            export_path = self.results_dir / f"{model_name}_retrieved_jobs_{self.timestamp}.json"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results exported to: {export_path}")
    
    def collect_user_inputs(self):
        """Collect all user inputs for query and evaluation parameters."""
        print(f"\n{'='*60}")
        print("USER INPUT COLLECTION")
        print(f"{'='*60}")
        
        # Get search query (use command line argument if provided, otherwise prompt)
        if self.args.query:
            query = self.args.query
            print(f"\nUsing query from command line: {query}")
        else:
            print("\nPlease enter your job search query:")
            query = input("Search query (e.g., 'Python Developer in San Francisco'): ").strip()
            if not query:
                query = "Python Developer in San Francisco with Salary 100000"
                print(f"Using default query: {query}")
        
        # Get expected location
        print("\nFor location accuracy testing, please specify the expected location:")
        expected_city = input("Expected city (e.g., 'San Francisco'): ").strip()
        expected_state = input("Expected state (e.g., 'CA'): ").strip()
        expected_country = input("Expected country (e.g., 'USA'): ").strip()
        
        # Get expected salary
        print("\nFor salary accuracy testing:")
        target_salary = input("Expected target salary (e.g., '100000'): ").strip() or "100000"
        
        # Get expected skills
        print("\nFor skill accuracy testing:")
        expected_skills_input = input("Expected skills (space-separated, e.g., 'python javascript react'): ").strip()
        expected_skills = expected_skills_input.split() if expected_skills_input else ["python"]
        
        return {
            'query': query,
            'expected_city': expected_city,
            'expected_state': expected_state,
            'expected_country': expected_country,
            'target_salary': target_salary,
            'expected_skills': expected_skills
        }

    def run_accuracy_tests(self, user_inputs):
        """Run accuracy tests on all models' results using provided user inputs."""
        print(f"\n{'='*60}")
        print("RUNNING ACCURACY TESTS")
        print(f"{'='*60}")
        
        # Extract user inputs
        expected_city = user_inputs['expected_city']
        expected_state = user_inputs['expected_state']
        expected_country = user_inputs['expected_country']
        target_salary = user_inputs['target_salary']
        expected_skills = user_inputs['expected_skills']
        query = user_inputs['query']
        
        accuracy_results = {}
        
        # Test scripts to run
        location_args = ["--query", query]
        if expected_city:
            location_args.extend(["--expected-city", expected_city])
        if expected_state:
            location_args.extend(["--expected-state", expected_state])  
        if expected_country:
            location_args.extend(["--expected-country", expected_country])
        
        # Prepare skill accuracy arguments with enhanced JSON output
        skill_args = ["--query", query, "--expected-skills"] + expected_skills
        
        test_scripts = [
            ("test_location_accuracy", "Location Accuracy", location_args),
            ("test_salary_accuracy", "Salary Accuracy", ["--query", query, "--target-salary", target_salary]),
            ("test_skill_accuracy", "Skill Accuracy", skill_args)
        ]
        
        for model_name in self.search_engines.keys():
            print(f"\n{'-'*50}")
            print(f"Testing {model_name}")
            print(f"{'-'*50}")
            
            model_results = {}
            results_file = self.results_dir / f"{model_name}_retrieved_jobs_{self.timestamp}.json"
            
            if not results_file.exists():
                print(f"Warning: Results file not found for {model_name}")
                continue
            
            for script_name, description, extra_args in test_scripts:
                print(f"\nRunning {description} test...")
                
                try:
                    # Prepare command
                    cmd = [
                        sys.executable, f"{script_name}.py",
                        "--retrieved-jobs-path", str(results_file)
                    ] + extra_args
                    
                    # Add enhanced JSON output path for skill accuracy test
                    if script_name == "test_skill_accuracy":
                        enhanced_json_path = self.results_dir / f"{model_name}_enhanced_skill_evaluation_{self.timestamp}.json"
                        cmd.extend(["--json-results-path", str(enhanced_json_path)])
                    
                    # Run the test script
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout
                    )
                    
                    if result.returncode == 0:
                        # Extract score from output (last line should be the score)
                        lines = result.stdout.strip().split('\n')
                        score = "N/A"
                        for line in reversed(lines):
                            line = line.strip()
                            if line and all(c.isdigit() or c in '.-' for c in line):
                                try:
                                    score = float(line)
                                    break
                                except ValueError:
                                    continue
                        
                        model_results[script_name] = {
                            "score": score,
                            "status": "success",
                            "output": result.stdout
                        }
                        print(f"✓ {description}: {score}")
                        
                    else:
                        model_results[script_name] = {
                            "score": "ERROR",
                            "status": "failed",
                            "error": result.stderr,
                            "output": result.stdout
                        }
                        print(f"✗ {description}: FAILED")
                        print(f"   Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    model_results[script_name] = {
                        "score": "TIMEOUT",
                        "status": "timeout",
                        "error": "Test timed out after 10 minutes"
                    }
                    print(f"✗ {description}: TIMEOUT")
                    
                except Exception as e:
                    model_results[script_name] = {
                        "score": "ERROR",
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"✗ {description}: ERROR - {e}")
                
                # Save individual test output
                if script_name in model_results:
                    output_file = self.results_dir / f"{model_name}_{script_name}_output_{self.timestamp}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"COMMAND: {' '.join(cmd)}\n")
                        f.write(f"RETURN CODE: {result.returncode if 'result' in locals() else 'N/A'}\n")
                        f.write(f"STDOUT:\n{model_results[script_name].get('output', 'N/A')}\n")
                        f.write(f"STDERR:\n{model_results[script_name].get('error', 'N/A')}\n")
            
            accuracy_results[model_name] = model_results
        
        return accuracy_results
    
    def generate_job_level_report(self, accuracy_results=None, user_query=None):
        """Generate a detailed job-level accuracy report for each model."""
        print(f"\n{'='*60}")
        print("GENERATING JOB-LEVEL ACCURACY REPORT")
        print(f"{'='*60}")
        
        query_to_use = user_query if user_query else self.args.query
        
        for model_name in self.search_engines.keys():
            print(f"\nProcessing {model_name}...")
            
            # Load retrieved jobs
            results_file = self.results_dir / f"{model_name}_retrieved_jobs_{self.timestamp}.json"
            if not results_file.exists():
                print(f"Warning: Results file not found for {model_name}")
                continue
                
            with open(results_file, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
                
            jobs = jobs_data.get('results', [])
            if not jobs:
                print(f"Warning: No jobs found for {model_name}")
                continue
            
            # Parse individual accuracy scores from test outputs
            location_scores = self._parse_individual_scores_from_output(model_name, "test_location_accuracy")
            salary_scores = self._parse_individual_scores_from_output(model_name, "test_salary_accuracy")
            skill_scores = self._parse_skill_scores_from_enhanced_json(model_name)
            
            # Create job-level report
            job_level_data = {
                "metadata": {
                    "timestamp": self.timestamp,
                    "query": query_to_use,
                    "model_name": model_name,
                    "total_jobs": len(jobs)
                },
                "jobs": []
            }
            
            for i, job in enumerate(jobs):
                job_entry = {
                    "job_id": job.get('job_id', i),
                    "rank": job.get('rank', i + 1),
                    "search_score": job.get('score', 0.0),
                    "job_title": job.get('job_title', ''),
                    "company_name": job.get('company_name', ''),
                    "location_raw": job.get('location_raw', ''),
                    "job_type": job.get('job_type', ''),
                    "salary_range": job.get('salary_range', ''),
                    "description": job.get('description', ''),
                    "accuracy_scores": {
                        "location_accuracy": location_scores.get(i, 0.0) if location_scores else 0.0,
                        "salary_accuracy": salary_scores.get(i, 0.0) if salary_scores else 0.0,
                        "skill_accuracy": skill_scores.get(i, 0.0) if skill_scores else 0.0
                    }
                }
                
                # Calculate overall accuracy score as average of the three
                accuracy_values = [
                    job_entry["accuracy_scores"]["location_accuracy"],
                    job_entry["accuracy_scores"]["salary_accuracy"],
                    job_entry["accuracy_scores"]["skill_accuracy"]
                ]
                job_entry["accuracy_scores"]["overall_accuracy"] = sum(accuracy_values) / len(accuracy_values)
                
                job_level_data["jobs"].append(job_entry)
            
            # Save job-level report
            job_level_file = self.results_dir / f"{model_name}_job_level_evaluation_{self.timestamp}.json"
            with open(job_level_file, 'w', encoding='utf-8') as f:
                json.dump(job_level_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Job-level report saved to: {job_level_file}")
            
            # Print summary statistics
            overall_scores = [job["accuracy_scores"]["overall_accuracy"] for job in job_level_data["jobs"]]
            if overall_scores:
                avg_overall = sum(overall_scores) / len(overall_scores)
                print(f"   Average overall accuracy: {avg_overall:.3f}")
                print(f"   Best job overall accuracy: {max(overall_scores):.3f}")
                print(f"   Jobs with >50% accuracy: {len([s for s in overall_scores if s > 0.5])}/{len(overall_scores)}")
        
        return job_level_file if 'job_level_file' in locals() else None
    
    def _parse_individual_scores_from_output(self, model_name: str, test_name: str) -> Dict[int, float]:
        """Parse individual job scores from test output files."""
        output_file = self.results_dir / f"{model_name}_{test_name}_output_{self.timestamp}.txt"
        
        if not output_file.exists():
            print(f"Warning: Output file not found: {output_file}")
            return {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            scores = {}
            lines = content.split('\n')
            
            # Method 1: Look for "Job X: 0.XX" format (used by salary accuracy)
            for line in lines:
                line = line.strip()
                if line.startswith('Job ') and ':' in line:
                    try:
                        job_part, score_part = line.split(':', 1)
                        job_num = int(job_part.replace('Job', '').strip()) - 1  # Convert to 0-based index
                        score = float(score_part.strip())
                        scores[job_num] = score
                    except (ValueError, IndexError):
                        continue
            
            # Method 2: If no scores found, look for "Individual Scores: [...]" format (used by location accuracy)
            if not scores:
                import re
                for line in lines:
                    line = line.strip()
                    if 'Individual Scores:' in line:
                        # Extract the list part: ['0.00', '0.50', '0.50', ...]
                        list_match = re.search(r'\[(.*?)\]', line)
                        if list_match:
                            scores_str = list_match.group(1)
                            # Split by comma and clean up each score
                            score_values = [s.strip().strip("'\"") for s in scores_str.split(',')]
                            for i, score_str in enumerate(score_values):
                                try:
                                    score = float(score_str)
                                    scores[i] = score
                                except ValueError:
                                    continue
                        break
            
            print(f"   Parsed {len(scores)} {test_name} scores")
            return scores
            
        except Exception as e:
            print(f"Error parsing {test_name} scores: {e}")
            return {}
    
    def _parse_skill_scores_from_enhanced_json(self, model_name: str) -> Dict[int, float]:
        """Parse individual job skill scores from enhanced skill evaluation JSON."""
        enhanced_file = self.results_dir / f"{model_name}_enhanced_skill_evaluation_{self.timestamp}.json"
        
        if not enhanced_file.exists():
            print(f"Warning: Enhanced skill file not found: {enhanced_file}")
            return {}
        
        try:
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = {}
            job_results = data.get('job_results', [])
            
            for job_result in job_results:
                job_id = job_result.get('job_id', 0)
                final_score = job_result.get('evaluation_scores', {}).get('final_score', 0.0)
                scores[job_id] = final_score
            
            print(f"   Parsed {len(scores)} skill accuracy scores")
            return scores
            
        except Exception as e:
            print(f"Error parsing skill scores: {e}")
            return {}
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        start_time = time.time()
        
        print("="*80)
        print("MULTI-MODEL JOB SEARCH EVALUATION WITH ACCURACY TESTING")
        print("="*80)
        print(f"Start time: {datetime.now()}")
        print(f"Results directory: {self.results_dir}")
        
        try:
            # Collect user inputs for query and evaluation parameters
            user_inputs = self.collect_user_inputs()
            
            # Load models
            self.load_models()
            
            # Run search evaluation with user-provided query
            self.run_search_evaluation(user_inputs['query'])
            
            # Run accuracy tests with user inputs
            accuracy_results = self.run_accuracy_tests(user_inputs)
            
            # Generate job-level accuracy report (updated to use user query)
            job_level_file = self.generate_job_level_report(accuracy_results, user_inputs['query'])
            
            elapsed_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("JOB-LEVEL EVALUATION COMPLETE")
            print(f"Total time: {elapsed_time:.2f} seconds")
            if job_level_file:
                print(f"Job-level report: {job_level_file}")
            
            # Show all generated files
            print(f"\nGenerated Files:")
            for model_name in self.search_engines.keys():
                print(f"  {model_name}:")
                
                # Search results
                search_file = self.results_dir / f"{model_name}_retrieved_jobs_{self.timestamp}.json"
                if search_file.exists():
                    print(f"    Search results: {search_file}")
                
                # Job-level evaluation
                job_level_file_model = self.results_dir / f"{model_name}_job_level_evaluation_{self.timestamp}.json"
                if job_level_file_model.exists():
                    print(f"    Job-level accuracy: {job_level_file_model}")
                
                # Enhanced skill evaluation
                enhanced_file = self.results_dir / f"{model_name}_enhanced_skill_evaluation_{self.timestamp}.json"
                if enhanced_file.exists():
                    print(f"    Enhanced skills: {enhanced_file}")
            
            print("="*80)
            
        except Exception as e:
            print(f"\nEvaluation failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Job Search Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model evaluation (with caching)
  python model_evaluation.py --model-1 ./nomic-bert-2048 --model-1-type sentence-transformer-recovered --jobs-data-path ./nomic-bert-2048/jobs.json
  
  # Compare two models
  python model_evaluation.py --model-1 ./nomic-bert-2048 --model-1-type sentence-transformer-recovered --model-2 ./other-model --model-2-type sentence-transformer --jobs-data-path ./jobs.json
  
  # Force recomputation of embeddings
  python model_evaluation.py --model-1 ./nomic-bert-2048 --model-1-type sentence-transformer-recovered --jobs-data-path ./nomic-bert-2048/jobs.json --force-recompute
  
  # Disable caching entirely
  python model_evaluation.py --model-1 ./nomic-bert-2048 --model-1-type sentence-transformer-recovered --jobs-data-path ./nomic-bert-2048/jobs.json --no-cache
        """
    )
    
    # Model 1 arguments
    parser.add_argument('--model-1', type=str, help='Path to first model directory')
    parser.add_argument('--model-1-type', type=str, choices=ModelLoader.SUPPORTED_TYPES,
                       help='Type of first model')
    
    # Model 2 arguments (optional for comparison)
    parser.add_argument('--model-2', type=str, help='Path to second model directory (optional)')
    parser.add_argument('--model-2-type', type=str, choices=ModelLoader.SUPPORTED_TYPES,
                       help='Type of second model (optional)')
    
    # Data arguments
    parser.add_argument('--jobs-data-path', type=str, required=True,
                       help='Path to jobs JSON data file')
    
    # Evaluation arguments
    parser.add_argument('--query', type=str, 
                       help='Search query to evaluate (will be collected interactively if not provided)')
    
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='Device to use for computation (default: auto)')
    
    # Caching arguments
    parser.add_argument('--cache-dir', type=str, default='embedding_cache',
                       help='Directory to store embedding cache files (default: embedding_cache)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of embeddings, ignoring cache')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable embedding caching completely')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.model_1 or not args.model_1_type:
        print("Error: --model-1 and --model-1-type are required")
        return 1
    
    if (args.model_2 and not args.model_2_type) or (not args.model_2 and args.model_2_type):
        print("Error: Both --model-2 and --model-2-type must be specified together")
        return 1
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        # Create evaluator and run evaluation
        evaluator = ModelEvaluator(args)
        evaluator.run_evaluation()
        return 0
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
