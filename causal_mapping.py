# filter and map on task decision graph

import re
import json
import numpy as np
import argparse
import asyncio
import time
from yacs.config import CfgNode
import sys
sys.path.append('../')
import utils as utils
from tqdm import tqdm
import pandas as pd
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from typing import Any, Dict

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)

causal_refine = """
You are a causality-aware recommender-system profile refiner.
Your task is to MAP and REFINE user attributes using a task decision graph.
Do NOT delete attributes. Do NOT invent new user facts.

You are given:
- a recommendation task,
- a causal decision graph for this task,
- a semantically merged user profile (flat list of attributes).

Your goal:
Refine the profile so that ALL attributes are:
(1) clearly mapped to decision steps,
(2) expressed in a task-actionable way,
(3) organized according to the decision graph.

IMPORTANT:
- Keep ALL attributes from the input profile.
- You may rewrite attributes for clarity, specificity, or task relevance.
- You may merge VERY similar attributes, but must not reduce information.

REFINEMENT PROCEDURE (FOLLOW EXACTLY):

Step 1 — Attribute → Decision Mapping
For EACH attribute:
- Identify the MOST relevant decision node in the graph.
- If an attribute is broadly applicable, assign it to the earliest relevant node.

Step 2 — Task-aware Attribute Refinement
For each attribute:
- Rewrite it (if needed) so it explicitly indicates HOW it influences
  decisions in this task (e.g., scoring, comparison, thresholding, tie-break).
- Make implicit preferences explicit (e.g., “likes X” → “prioritises X when ranking”).

Step 3 — Intra-node Consolidation (Light)
Within each decision node:
- If multiple attributes express the SAME control signal, merge them into
  ONE clearer attribute.
- Preserve nuances; do not over-compress.

Do NOT fabricate.
Do NOT keep vague personality traits.
Do NOT keep descriptive but not decision-affecting,

INPUT:

TASK NAME:
{task_name} {task_description}

TASK DOMAIN: {item_name} recommendations

TASK EXAMPLE (Help understand the task):
{ICL_example}

DECISION GRAPH (Causal):
{decision_graph}

SEMANTIC CORE PROFILE:
{semantic_core_profile}

OUTPUT FORMAT (STRICT):
- Keep profile concise and actionable.
- No evaluation or comments. 
- Output ONLY the final refined attributes as bullet points.

"""


causal_refine_test = """
You are a causality-aware recommender-system profile refiner.
Your task is to MAP and REFINE user attributes using a task decision graph.
Do NOT delete attributes. Do NOT invent new user facts.

You are given:
- a recommendation task,
- a causal decision graph for this task,
- a semantically merged user profile (flat list of attributes).

Your goal:
Refine the profile so that ALL attributes are:
(1) clearly mapped to decision steps,
(2) expressed in a task-actionable way.

IMPORTANT:
- Keep ALL attributes from the input profile.
- You may rewrite attributes for clarity, specificity, or task relevance.
- You may merge VERY similar attributes, but must not reduce information.

REFINEMENT PROCEDURE (FOLLOW EXACTLY):

Step 1 — Attribute → Decision Mapping
Map each user attribute to the decision graph node(s) it can directly change by asking whether modifying only that attribute would alter the decision made at that node. If an attribute does not affect any outcome-changing decision, do not map it. One attribute may map to multiple nodes if it causally influences more than one decision.
For EACH attribute:
- Identify the MOST relevant decision node in the graph.
- If an attribute is broadly applicable, assign it to the earliest relevant node.

Step 2 — Task-aware Attribute Refinement
For each attribute:
- Refinement MUST be guided by the decision graph.
- Ask: “Can this attribute, AS WRITTEN, directly influence the assigned
  decision node?”
    - If YES, keep the original attribute UNCHANGED.
    - If NO, refine the attribute based on decision graph, while preserving its original meaning.

Step 3 — Intra-node Consolidation (Light)
Within each decision node:
- If multiple attributes express the SAME control signal, merge them into ONE clearer attribute.
- Preserve nuances; do not over-compress.

Do NOT fabricate.

INPUT:

TASK NAME:
{task_name} {item_name} recommendations. {task_description}

TASK EXAMPLE (Help understand the task):
{ICL_example}

DECISION GRAPH (Causal):
{decision_graph}

SEMANTIC CORE PROFILE:
{semantic_core_profile}

OUTPUT FORMAT (STRICT):
- Keep profile concise and actionable.
- No evaluation or comments. 
- Output ONLY the final refined attributes as bullet points.

OUTPUT (STRICT):
Return profile in following format:
  Taste: 
  <brief content> 
  <brief content>
  ...
  
  Behaviour Pattern: 
  <brief content>
  <brief content>
  ...
"""

causal_refine_2 = """
You are a causal reasoning expert for recommendation simulation.

You are given:
1) a recommendation task name and description,
2) one concrete task example (including item features and expected output),
3) a causal decision graph (ordered, outcome-changing decisions),
4) a semantically merged user profile to be refined (unified attribute names with values).

Your goal:
Refine the user profiles using causal reasoning optimised for the given task.

GENERAL PRINCIPLES:
- Focus on causal influence, not correlation or evaluation logic.
- Use only user attributes and item features provided in the task example.
- Never reference ground truth, labels, observed interactions, or correct outputs.
- Keep only information that can change the task outcome.

CAUSAL REFINEMENT PROCEDURE (ADAPTIVE):

1. Task Understanding.
2. Causal Mapping:
Map each profile attribute to the decision graph node(s) it can directly influence.
3. Causal Relevance Test (MANDATORY):
For each attribute, ask: “If this attribute alone were changed, would the decision of the task change?”
   Discard attributes that fail this test.
4. Minimal Sufficiency Refinement:
If multiple attributes affect the same decision or outcome, keep the smallest subset that preserves the same task outcome.
5. Priority Allocation:
   Keep more attributes for high-impact decision nodes and fewer for low-impact or tie-breaking nodes.
6. Finalisation:
   Output the refined profile, grouping attributes by the decision node(s) they support.

INPUT:

TASK NAME:
{task_name} in {item_name} recommendations. {task_description}

TASK EXAMPLE:
{ICL_example}

DECISION GRAPH (Causal):
{decision_graph}

SEMANTIC PROFILE:
{semantic_core_profile}

VALIDITY SELF-CHECK (before output):
- Every kept attribute maps to at least one decision node.
- No discarded attribute can change any decision node.
- The final profile is minimal but sufficient for the task.

OUTPUT FORMAT (STRICT):
- Final Refined Profile with bullet points attributes (grouped by causal role)
- No extra comments or profile irrelevant text.

"""

causal_refine_1 = """
You are a causal reasoning expert for recommendation simulation.

You are given:
1) a recommendation task name and description,
2) one concrete task example (including item features and expected output),
3) a causal decision graph (ordered, outcome-changing decisions),
4) a semantically merged user profile to be refined (unified attribute names with values).

Your goal:
Refine the user profiles using causal reasoning optimised for the given task.

GENERAL PRINCIPLES:
- Focus on causal influence, not correlation or evaluation logic.
- Use only user attributes and item features provided in the task example.
- Never reference ground truth, labels, observed interactions, or correct outputs.
- Keep only information that can change the task outcome.

CAUSAL REFINEMENT PROCEDURE (ADAPTIVE):

1. Task Understanding.
2. Causal Grouping:
   - If using a decision graph: group attributes by the decision node(s) they influence.
   - If no decision graph: group attributes by outcome role (e.g., preference signal, avoidance signal, intensity calibration, tie-breaker).
3. Priority Allocation:
   Keep more attributes for high-impact decision nodes and fewer for low-impact or tie-breaking nodes.

INPUT:

TASK NAME:
{task_name} in {item_name} recommendations. {task_description}

TASK EXAMPLE:
{ICL_example}

DECISION GRAPH (Causal):
{decision_graph}

SEMANTIC PROFILE:
{semantic_core_profile}

VALIDITY SELF-CHECK (before output):
- The final profile is minimal but sufficient for the task.

OUTPUT FORMAT (STRICT):
- Final Refined Profile with bullet points attributes (grouped by causal role)
- No extra comments or profile irrelevant text.

"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/config_general.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-p", "--profile_file", type=str, default="", help="Path to output profile file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default="output/version3/gen_profiles.csv", help="Path to output profile file"
    )
    parser.add_argument(
        "-n", "--profile_num", type=int, default=3
    )
    parser.add_argument(
        "-g", "--gen_decision_graph", type=bool, default=False
    )
    args = parser.parse_args()
    return args

async def main(args):
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    utils.fix_seeds(config.seed) # set random seed
    # load llm
    config.temperature = 0.1
    if 'gpt' in config.llm and '5' in config.llm:
        config.temperature = 1
    llm = utils.load_LLM(config)
    
    # data preparation
    if 'book' in args.output_file.lower():
        print('load book')
        from data.dataloader_new import Amazon_Book
        dataloader = Amazon_Book(config)
        item_name = 'book'
        with open('decision_graph/amazon_book_test.json', 'r') as f:
            tasks_list = json.load(f)
        print(f"Load amazon-book\n")
    elif 'beauty' in args.output_file.lower():
        print('load beauty')
        from data.dataloader_new import Amazon_Beauty
        dataloader = Amazon_Beauty(config)
        item_name = 'beauty product'
        with open('decision_graph/amazon_beauty.json', 'r') as f:
            tasks_list = json.load(f)
        print(f"Load amazon-beauty\n")
    else:
        from data.dataloader_new import ML_1M
        print('load ml_1m')
        dataloader = ML_1M(config)
        item_name = 'movie'
        with open('decision_graph/ml_1m_test.json', 'r') as f:
            tasks_list = json.load(f)
        print(f"Load movielens\n")
    
    profiles_df = pd.read_csv(args.profile_file)
    profiles_df = profiles_df.set_index('userid')
    profiles_dict = profiles_df.to_dict(orient='index')
    print(f"All users:{profiles_dict.keys()}")
    
    if args.gen_decision_graph:
        prompt_dict = utils.load_prompt("prompt/task_aware.yaml")
        positive_items, negative_items = dataloader.get_items_for_profile(userid=list(profiles_dict.keys())[0], pos_num=3, neg_num = 3)
        for i, task_info in enumerate(tasks_list):
            task_specific_suggestions = await decision_graph_extraction(prompt_dict, llm, item_name, task_info, positive_items[:3], negative_items[:3])
            tasks_list[i]['decision_graph'] = task_specific_suggestions
        # print(tasks_list)

    avatars_info = {}
    for userid in tqdm(profiles_dict.keys()):
        positive_items, negative_items = dataloader.get_items_for_profile(userid=userid, pos_num=3, neg_num = 3)
        # profiles_list = await profile_selection(profiles_list, positive_items)
        avatars_info[userid] = {}
        for task_info in [tasks_list[0], tasks_list[2]]:
            name = task_info['action_name']+'_profile'
            if 'disc' in task_info['action_name']:
                name = 'discrimination_profile'
            if 'semantic_profile' in profiles_dict[userid].keys():
                semantic_profile = profiles_dict[userid]['semantic_profile']
            else:
                semantic_profile = profiles_dict[userid][name]
            merged_profile = await causal_merge(llm, semantic_profile, item_name, task_info, positive_items, negative_items)
            if len(avatars_info) == 1:
                print(name, '\n')
                print(merged_profile)
            avatars_info[userid][name]= merged_profile
    df = pd.DataFrame.from_dict(avatars_info, orient='index')
    df.index.name = 'userid'
    df = df.reset_index()

    try:
        lt = time.strftime('%m-%d_%H:%M', time.localtime())
        output_path = args.output_file.replace('.csv', '_'+str(lt)+'.csv')
        df.to_csv(output_path, index=False)
        print('save at: ', output_path)
    except:
        df.to_csv(f"output/{item_name}_{lt}_task_aware_profile.csv", index=False)
        
        print(f"save at: /Users/wenny/Documents/study/PhD/simulation_experiment/AutoProfileGenerator/output/{item_name}_{lt}_task_aware_profile.csv")
    return 1


async def causal_merge(llm, semantic_core_profile, item_name, task_info, positive_items, negative_items):
    # causal_refine
    # causal_refine_test
    # causal_refine_2
    messages = PromptTemplate(causal_refine_test).format_messages(
        item_name = item_name,
        task_name = task_info['task_name'],
        task_description = task_info['task_description'],
        action_name = task_info['action_name'],
        decision_graph = task_info['decision_graph'],
        ICL_example = utils.format_ICL_examples(positive_items, negative_items, task_info['task_name']),
        semantic_core_profile = semantic_core_profile
    )
    raw_rating_based_response = await llm.achat(messages = messages)
    merged_profile = raw_rating_based_response.message.content
    
    return merged_profile        

async def decision_graph_extraction(prompt_dict, llm, item_name, task_info, positive_items, negative_items):
    # print(task_info)task_decision_graph_causal
    message = PromptTemplate(prompt_dict['task_decision_graph_extraction']).format_messages(
        item_name = item_name,
        task_name = task_info['task_name'],
        task_description = task_info['task_description'],
        interaction_name = task_info['action_name'],
        ICL_example = utils.format_ICL_examples(positive_items, negative_items, task_info['task_name']),
        # items_list = utils.recommendation_list_to_prompt(example_items)
    )
    raw_rating_based_response = await llm.achat(messages = message)
    extracted_features = raw_rating_based_response.message.content
    print(task_info['action_name'])
    print(extracted_features)
    print('\n')
    return extracted_features

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))