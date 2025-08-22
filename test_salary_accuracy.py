import os
import json
import statistics
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import pprint
import re
import argparse
import sys
import io

# Force UTF-8 stdout (fixes charmap issues on Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"
DEFAULT_RETRIEVED_JOBS_PATH = "retrieved_jobs.json"  # ← Default JSON file


# Initialize DeepSeek Client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

class SalaryTester:
    def __init__(self, retrieved_jobs_path=None):
        self.pp = pprint.PrettyPrinter(indent=2)
        self.retrieved_jobs_path = retrieved_jobs_path or DEFAULT_RETRIEVED_JOBS_PATH

    def load_jobs_from_json(self, query: str) -> Dict:
        """Load jobs from retrieved_jobs.json instead of calling HireBase API"""
        try:
            print(f"\nLoading retrieved jobs from: {self.retrieved_jobs_path}")
            with open(self.retrieved_jobs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Adjust to handle actual format
            if isinstance(data, dict):
                if "results" in data:
                    jobs_list = data["results"]
                elif "jobs" in data:
                    jobs_list = data["jobs"]
                else:
                    raise ValueError("No 'results' or 'jobs' key found in retrieved_jobs.json.")
            elif isinstance(data, list):
                jobs_list = data
            else:
                raise ValueError("Unrecognized JSON structure")

            print(f"Loaded {len(jobs_list)} jobs from JSON file.")
            return {"jobs": jobs_list}
        except Exception as e:
            print(f"Error loading jobs: {e}")
            return {"error": str(e)}


    def call_deepseek_llm(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Call DeepSeek LLM for salary evaluation"""
        try:
            print("Analyzing results with LLM...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                stream=False,
                timeout=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return f"ERROR: {e}"

    def format_jobs_for_evaluation(self, jobs: List[Dict]) -> str:
        """Return raw job JSON (first 10), with HTML tags removed from description."""
        if not jobs:
            return "No jobs found."

        cleaned_jobs = []
        for job in jobs:
            job_copy = dict(job)
            description = job_copy.get('description')
            if isinstance(description, str):
                job_copy['description'] = re.sub(r'<[^>]+>', '', description)
            cleaned_jobs.append(job_copy)

        return json.dumps(cleaned_jobs, separators=(',', ':'))

    def create_evaluation_prompt(self, query: str, target_salary: int, formatted_jobs: str) -> List[Dict]:
        """Create the prompt for LLM salary evaluation"""
        min_threshold = int(target_salary * 0.75)  # 75% of target
        
        system_prompt = f"""You are an expert at evaluating job search results for salary accuracy. You will score jobs based on how well their salary ranges match the target salary expectation.

SCORING SYSTEM (per job):
- Target salary: ${target_salary:,} USD per year
- Minimum threshold: ${min_threshold:,} USD per year (75% of target)

SCORING RULES:
1. If job salary >= ${target_salary:,}: Score = 1.0
2. If job salary between ${min_threshold:,} and ${target_salary:,}: Score = linear scale from 0.0 to 1.0
3. If job salary < ${min_threshold:,}: Score = 0.0
4. If no salary information provided: Score = 0.0
5. For salary ranges, use the minimum of the range for scoring
6. Convert all salaries to USD per year equivalent for comparison
7. Return scores in exact format: Job X: 0.XX

RESPONSE FORMAT:
Job 1: 0.XX
Job 2: 0.XX
...
Job 10: 0.XX

FINAL_AVERAGE: X.XX
HIT_RATE: X.XX (percentage of jobs with score >= 0.6)"""

        expected_str = f"Target salary: ${target_salary:,} USD per year (anything above this gets 1.0, linear scale from 75% to 100%)"
        
        user_prompt = f"""Query: "{query}"
{expected_str}

Job Results to Evaluate:
{formatted_jobs}

Please evaluate each job's salary accuracy using the scoring system. Focus on the first 10 jobs only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def parse_llm_scores(self, llm_response: str) -> Tuple[List[float], float, float]:
        """Parse LLM response to extract individual scores, average, and hit rate"""

        print(llm_response)

        scores = []
        average_score = 0.0
        hit_rate = 0.0
        
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse individual job scores
            if line.startswith('Job ') and ':' in line:
                try:
                    score_part = line.split(':')[1].strip()
                    score = float(score_part)
                    scores.append(score)
                except (ValueError, IndexError):
                    continue
            
            # Parse final average
            elif line.startswith('FINAL_AVERAGE:'):
                try:
                    average_score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            
            # Parse hit rate
            elif line.startswith('HIT_RATE:'):
                try:
                    hit_rate = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
        
        # Calculate average if not provided
        if not average_score and scores:
            average_score = statistics.mean(scores)
        
        # Calculate hit rate if not provided
        if not hit_rate and scores:
            high_quality_jobs = len([s for s in scores if s >= 0.6])
            hit_rate = (high_quality_jobs / len(scores)) * 100 if scores else 0
        
        return scores, average_score, hit_rate

    def get_user_input(self) -> Tuple[str, int]:
        """Get query and target salary from environment variables"""
        print("Salary Accuracy Testing Tool")
        print("=" * 50)
        print("\nFetching job search query and target salary from environment variables...")
        print("Note: Assumes USD per year. Anything above target gets 1.0 score.")

        # Get query
        query = os.getenv("JOB_QUERY", "").strip()
        if not query:
            raise ValueError("Environment variable JOB_QUERY is required and cannot be empty.")

        # Get and validate target salary
        salary_str = os.getenv("TARGET_SALARY", "").replace(",", "").strip()
        try:
            target_salary = int(salary_str)
            if target_salary <= 0:
                raise ValueError("TARGET_SALARY must be a positive integer.")
        except ValueError:
            raise ValueError("Environment variable TARGET_SALARY must be a valid positive integer.")

        return query, target_salary

    def evaluate_single_query(self, query: str, target_salary: int) -> float:
        """Evaluate salary accuracy for a single user query"""
        print(f"\n{'='*80}")
        print(f"Testing Query: \"{query}\"")
        print(f"Target Salary: ${target_salary:,} USD/year (1.0 score at this level and above)")
        print(f"Minimum Threshold: ${int(target_salary * 0.75):,} USD/year (0.0 score below this)")
        print(f"{'='*80}")
        
        # Load jobs from JSON file
        api_response = self.load_jobs_from_json(query)

        jobs = api_response.get('jobs', [])

        total_jobs = len(jobs)
        print(f"Found {total_jobs} jobs, evaluating first 10...")
        
        # Format jobs for evaluation
        formatted_jobs = self.format_jobs_for_evaluation(jobs)
        
        # Create evaluation prompt
        evaluation_messages = self.create_evaluation_prompt(
            query, 
            target_salary, 
            formatted_jobs
        )
        
        # Get LLM evaluation
        llm_response = self.call_deepseek_llm(evaluation_messages)
        
        if llm_response.startswith("ERROR:"):
            print(f"LLM Error: {llm_response}")
            return 0.0
        
        # Parse scores
        individual_scores, average_score, hit_rate = self.parse_llm_scores(llm_response)
        
        # Print results
        print(f"\nResults:")
        print(f"   Average Score: {average_score:.2f}")
        print(f"   Hit Rate: {hit_rate:.1f}% (jobs with score >= 0.6)")
        print(f"   Jobs Evaluated: {len(individual_scores)}")
        print(f"   Individual Scores: {[f'{s:.2f}' for s in individual_scores]}")
        
        return average_score

def main():
    """Main function to run interactive salary accuracy test"""
    parser = argparse.ArgumentParser(description="Test salary accuracy of job search results")
    parser.add_argument('--retrieved-jobs-path', type=str, default=DEFAULT_RETRIEVED_JOBS_PATH,
                       help='Path to retrieved jobs JSON file')
    parser.add_argument('--query', type=str, help='Search query to evaluate')
    parser.add_argument('--target-salary', type=int, help='Target salary for the query')
    
    args = parser.parse_args()
    
    tester = SalaryTester(args.retrieved_jobs_path)
    
    try:
        if args.query and args.target_salary:
            # Use provided arguments
            query = args.query
            target_salary = args.target_salary
        else:
            # Get user input
            query, target_salary = tester.get_user_input()
        
        # Evaluate the query
        score = tester.evaluate_single_query(query, target_salary)
        
        # Output final result
        print(f"\n{'='*50}")
        print(f"FINAL RESULT")
        print(f"{'='*50}")
        print(f"Salary Accuracy Score: {score:.2f}")
        print(f"{'='*50}")
        
        # Also print just the number for easy extraction
        print(f"\n{score:.2f}")
        
        return score
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return 0.0
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("0.00")  # Return 0 score on error
        return 0.0

if __name__ == "__main__":
    main() 