import traceback
from litellm import completion
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



takeaway_eval_prompt = """You are an expert evaluator, evaluating how effective a research assistant is. The research assistant is tasked with identifying insightful takeaways from different domains that are meaningful for addressing the research problem. You will assess multiple of these takeaways based on the provided criteria, and rank them in order of quality.

# Research Problem:
{research_problem}

# Takeaway 1:
{takeaway_1}

# Takeaway 2:
{takeaway_2}

# Criteria for Evaluation:
- Best rational
- Best potential for integrating well with target domains
- Most surprising and interesting

# Output Format:
{{
   "reasoning": str, # Brief reasoning (2-3 sentences max). Explain your reasoning for evaluating the takeaway.
   "ranking": list[int], # List of integers representing the order of prefered takeaways. It should be [1, 2] if the first takeaway is preferred over the second takeaway, and [2, 1] if the second takeaway is preferred over the first takeaway.
}}"""

idea_eval_prompt = """You are an expert evaluator, evaluating how effective a research assistant is. The research assistant is proposing an idea for a research problem. You will assess multiple of these ideas based on the provided criteria, and rank them in order of quality.

# Research Problem:
{research_problem}

# Idea 1:
{idea_1}

# Idea 2:
{idea_2}

# Criteria for Ideas:
- Best rational
- Best potential for integrating well with target domains
- Most surprising and interesting

# Output Format:
{{
   "reasoning": str, # Brief reasoning (2-3 sentences max). Explain your reasoning for evaluating the takeaway.
   "ranking": list[int], # List of integers representing the order of prefered takeaways. It should be [1, 2] if the first takeaway is preferred over the second takeaway, and [2, 1] if the second takeaway is preferred over the first takeaway.
}}"""


class Evaluator:
    def __init__(
        self,
        num_retries=10,
        model_name=None,
        api_base=None,
        api_key=None
    ):
        self.num_retries = num_retries

        self.model_name = model_name
        self.kwargs = {"temperature": 0.0, "max_tokens": 1024}
        if api_base and api_key:
            self.kwargs["api_base"] = api_base
            self.kwargs["api_key"] = api_key

    def completion(self, messages):
        return completion(model=self.model_name, messages=messages, num_retries=self.num_retries, **self.kwargs).choices[0].message.content

    def evaluate_sample(self, sample):
        eval_results = {}
        research_problem,selected_takeaways,gt_takeaways,proposed_idea,gt_idea = sample,sample,sample,sample,sample

        # Part 1: Evaluate takeaways
        for _ in self.num_retries:
            processed_takeaway_eval_prompt = takeaway_eval_prompt.format(
                research_problem=research_problem,
                takeaway_1=selected_takeaways,
                takeaway_2=gt_takeaways
            )

            messages = [{"role": "user", "content": processed_takeaway_eval_prompt}]
            takeaway_eval_response = self.completion(messages)
            takeaway_eval_response = repair_json(takeaway_eval_response, return_objects=True)

            if "reasoning" in takeaway_eval_response and "ranking" in takeaway_eval_response:
                if takeaway_eval_response["ranking"] == [1, 2]:
                    eval_results["takeaways"] = {
                        "judge_response": takeaway_eval_response,
                        "selected_takeaway_win": True
                    }
                    break
                elif takeaway_eval_response["ranking"] == [2, 1]:
                    eval_results["takeaways"] = {
                        "judge_response": takeaway_eval_response,
                        "selected_takeaway_win": False
                    }
                    break

        # Part 2: Evaluate overall proposed idea
        for _ in self.num_retries:
            processed_idea_eval_prompt = idea_eval_prompt.format(
                research_problem=research_problem,
                idea_1=proposed_idea,
                idea_2=gt_idea
            )

            messages = [{"role": "user", "content": processed_idea_eval_prompt}]
            idea_eval_response = self.completion(messages)
            idea_eval_response = repair_json(idea_eval_response, return_objects=True)

            if "reasoning" in idea_eval_response and "ranking" in idea_eval_response:
                if idea_eval_response["ranking"] == [1, 2]:
                    eval_results["idea"] = {
                        "judge_response": idea_eval_response,
                        "selected_idea_win": True
                    }
                    break
                elif takeaway_eval_response["ranking"] == [2, 1]:
                    eval_results["idea"] = {
                        "judge_response": idea_eval_response,
                        "selected_idea_win": False
                    }
                    break

        return eval_results

    def evaluate_samples(self, samples):
        evaluated_samples = []
        batch_size = 300
        total_successes,total_failures = len(evaluated_samples),0

        with tqdm(total=len(samples), desc="Evaluating samples") as progress_bar:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as executor:
                    futures_to_index = {executor.submit(self.evaluate_sample, sample): sample for sample in batch}

                    for future in as_completed(futures_to_index):
                        curr_result = future.result()

                        if curr_result == None:
                            total_failures += 1
                            continue                        
                        evaluated_samples.append(curr_result)

                        total_successes += 1
                        progress_bar.update(1)




        print(f"\n\n\nEvaluation complete!")
        print(f"    # succeeded conversations: {total_successes}")
        print(f"    # failed conversations: {total_failures}")

        # Print summary statistics
        takeaway_win_rate = sum([1 for sample in evaluated_samples if sample['takeaways']['selected_takeaway_win']]) / len(evaluated_samples)
        idea_win_rate = sum([1 for sample in evaluated_samples if sample['idea']['selected_idea_win']]) / len(evaluated_samples)
        print(f"    # takeaway win rate: {takeaway_win_rate}")
        print(f"    # idea win rate: {idea_win_rate}")

        return evaluated_samples