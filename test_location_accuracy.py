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
DEFAULT_RETRIEVED_JOBS_PATH = "retrieved_jobs.json"  # ← Default JSON file
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# Initialize DeepSeek Client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

class LocationTester:
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
        """Call DeepSeek LLM for location evaluation"""
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

    def create_evaluation_prompt(self, query: str, expected_location: Dict, formatted_jobs: str) -> List[Dict]:
        """Create the prompt for LLM location evaluation"""
        system_prompt = """You are an expert at evaluating job search results for location accuracy. You will score jobs based on how well their locations match the expected location from a search query.

SCORING SYSTEM (per job):
- City/Region match: 0.33 points (if the job location city matches expected city or is in the same metropolitan area)
- State/Province match: 0.33 points (if the job location state/province matches expected). You can assume/extrapolate the state/province from the city.
- Country match: 0.33 points (if the job location country matches expected). You can assume/extrapolate the country from the state/province.

IMPORTANT RULES:
1. Score each job individually from 0.0 to 1.0 (sum of the three components)
2. If expected city is None, ignore city scoring and redistribute points (0.5 for state, 0.5 for country)
3. If expected state is None, ignore state scoring and redistribute points (0.5 for city, 0.5 for country)
4. Consider metropolitan areas (e.g., "San Francisco" includes "South San Francisco", "San Mateo", etc.)
5. Be lenient with state/province abbreviations and full names
6. Return scores in exact format: Job X: 0.XX

RESPONSE FORMAT:
Job 1: 0.XX
Job 2: 0.XX
...
Job 10: 0.XX

FINAL_AVERAGE: X.XX
HIT_RATE: X.XX (percentage of jobs with score >= 0.67)"""

        expected_str = f"Expected location - City: {expected_location.get('city', 'Any')}, State: {expected_location.get('state', 'Any')}, Country: {expected_location.get('country', 'Any')}"
        
        user_prompt = f"""Query: "{query}"
{expected_str}

Job Results to Evaluate:
{formatted_jobs}

Please evaluate each job's location accuracy using the scoring system. Focus on the first 10 jobs only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def parse_llm_scores(self, llm_response: str) -> Tuple[List[float], float, float]:
        """Parse LLM response to extract individual scores, average, and hit rate"""
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
            high_quality_jobs = len([s for s in scores if s >= 0.67])
            hit_rate = (high_quality_jobs / len(scores)) * 100 if scores else 0
        
        return scores, average_score, hit_rate

    def get_user_input(self) -> Tuple[str, Dict[str, str]]:
        """Get query and expected location from user input"""
        print("Location Accuracy Testing Tool")
        print("=" * 50)
        print("\nInteractive mode - please provide query and location details...")

        # Get query from user input
        query = input("\nEnter job search query: ").strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        # Get location details from user input
        print("\nEnter expected location details (press Enter to skip):")
        city = input("Expected city: ").strip() or None
        state = input("Expected state: ").strip() or None  
        country = input("Expected country: ").strip() or None

        expected_location = {"city": city, "state": state, "country": country}
        if not any(expected_location.values()):
            raise ValueError("At least one location field (city, state, or country) must be provided.")

        return query, expected_location

    def evaluate_single_query(self, query: str, expected_location: Dict[str, str]) -> float:
        """Evaluate location accuracy for a single user query"""
        print(f"\n{'='*80}")
        print(f"Testing Query: \"{query}\"")
        print(f"Expected: {expected_location}")
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
            expected_location, 
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
        print(f"   Hit Rate: {hit_rate:.1f}% (jobs with score ≥ 0.67)")
        print(f"   Jobs Evaluated: {len(individual_scores)}")
        print(f"   Individual Scores: {[f'{s:.2f}' for s in individual_scores]}")
        
        return average_score

def main():
    """Main function to run interactive location accuracy test"""
    parser = argparse.ArgumentParser(description="Test location accuracy of job search results")
    parser.add_argument('--retrieved-jobs-path', type=str, default=DEFAULT_RETRIEVED_JOBS_PATH,
                       help='Path to retrieved jobs JSON file')
    parser.add_argument('--query', type=str, help='Search query to evaluate')
    parser.add_argument('--expected-city', type=str, help='Expected city for the query')
    parser.add_argument('--expected-state', type=str, help='Expected state for the query')
    parser.add_argument('--expected-country', type=str, help='Expected country for the query')
    # Keep backward compatibility
    parser.add_argument('--expected-location', type=str, help='Expected location for the query (backward compatibility - treated as city)')
    
    args = parser.parse_args()
    
    tester = LocationTester(args.retrieved_jobs_path)
    
    try:
        if args.query and (args.expected_city or args.expected_state or args.expected_country or args.expected_location):
            # Use provided arguments
            query = args.query
            
            # Build expected_location from individual arguments
            expected_location = {
                "city": args.expected_city,
                "state": args.expected_state, 
                "country": args.expected_country
            }
            
            # Handle backward compatibility for --expected-location (treat as city)
            if args.expected_location and not args.expected_city:
                expected_location["city"] = args.expected_location
                
        else:
            # Get user input
            query, expected_location = tester.get_user_input()
        
        # Evaluate the query
        score = tester.evaluate_single_query(query, expected_location)
        
        # Output final result
        print(f"\n{'='*50}")
        print(f"FINAL RESULT")
        print(f"{'='*50}")
        print(f"Location Accuracy Score: {score:.2f}")
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