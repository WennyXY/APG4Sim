import re
import random
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/config_general.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default="output/version3/gen_profiles.csv", help="Path to output profile file"
    )
    parser.add_argument(
        "-n", "--initial_num", type=int, default=3
    )
    parser.add_argument(
        "-in", "--interaction_num", type=int, default=15
    )
    args = parser.parse_args()
    return args


async def main(args):
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    utils.fix_seeds(config.seed) # set random seed
    
    lt = time.strftime('%m-%d %H:%M', time.localtime())
    args.output_file = args.output_file.replace('.csv', '_'+str(lt)+'.csv')
    # load llm
    config.temperature = 0.1
    client = utils.load_client(config)
    
    # data preparation
    if 'book' in args.output_file.lower():
        from data.dataloader_new import Amazon_Book
        dataloader = Amazon_Book(config)
        item_name = 'book'
        print(f"Load amazon-book\n")
    elif 'beauty' in args.output_file.lower():
        from data.dataloader_new import Amazon_Beauty
        dataloader = Amazon_Beauty(config)
        item_name = 'beauty product'
        print(f"Load amazon-beauty\n")
    else:
        from data.dataloader_new import ML_1M
        dataloader = ML_1M(config)
        item_name = 'movie'
        print(f"Load movielens\n")

    prompt_dict = utils.load_prompt("prompt/task_aware.yaml")
    profile_generator = TaskAwareProfileGeneration(llm=None, client = client, config={'model':config.llm.replace('azure:',''), 'temperature':config.temperature}, prompt_dict=prompt_dict, initialisation_num=args.initial_num, item_name=item_name, timeout=1000)
    profile_num = 20
    userid_list = dataloader.get_users_df()['userid'].tolist()
    userid_list = np.random.choice(userid_list, min(len(userid_list), profile_num), replace=False)
    
    print(f"All users:{userid_list}")
    avatars_info = {}
    for userid in tqdm(userid_list):
        positive_items, negative_items = dataloader.get_items_for_profile(userid=userid, pos_num=args.interaction_num, neg_num = 10)
        dataset_context = '; '.join(negative_items[0].keys())
        
        profiles_list = await profile_generator.run(positive_items=positive_items, negative_items=negative_items, dataset_context=dataset_context)

        avatars_info[userid] = {}
        for i, profile_score in enumerate(profiles_list):
            avatars_info[userid]['init_profile'+str(i)] = profile_score['profile']
    df = pd.DataFrame.from_dict(avatars_info, orient='index')
    df.index.name = 'userid'
    df = df.reset_index()

    try:
        df.to_csv(args.output_file, index=False)
        print('save at: ',args.output_file)
    except:
        df.to_csv(f"output/{item_name}_{lt}_task_aware_profile.csv", index=False)
        
        print(f"save at: output/{item_name}_{lt}_task_aware_profile.csv")
    return 1


class ProfileGenerationEvent(Event):
    initialised_profile: str
    rating_score: float

class ProfileInitialisationEvent(Event):
    positive_items: list
    dataset_context: str

class ProfileMergingEvent(Event):
    positive_items: list
    dataset_context: str

class TaskAwareProfileGeneration(Workflow):
    def __init__(self, llm: LLM|None=None, client: LLM|None=None, config: Dict ={}, prompt_dict: Dict ={}, initialisation_num=3, item_name='movie', **workflow_kwargs: Any,)-> None:
        super().__init__(**workflow_kwargs)
        self.llm = llm
        self.client = client
        self.config = config
        self.item_name = item_name
        self.prompt_dict = prompt_dict
        self.initialisation_num = initialisation_num
        print(f"item_name:{item_name} initialisation_num:{initialisation_num}")

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> ProfileInitialisationEvent:
        dataset_context = ev.dataset_context
        positive_items = ev.positive_items
        negative_items = ev.negative_items
        await ctx.store.set("dataset_context", dataset_context)
        await ctx.store.set("positive_items", positive_items)
        await ctx.store.set("negative_items", negative_items)
        print('start initialising')
        for i in range(self.initialisation_num):
            ctx.send_event(ProfileInitialisationEvent(positive_items = ev.positive_items, dataset_context = dataset_context))
        return None
    
    @step
    async def profile_generation(self, ctx: Context, ev: ProfileGenerationEvent) -> StopEvent:
        results = ctx.collect_events(ev, [ProfileGenerationEvent] * self.initialisation_num)
        
        if results is None:
            return None
        
        profile_score_pairs = []
        for event in results:
            profile_score_pairs.append({
                'profile': event.initialised_profile,
                'rating_score': event.rating_score
            })
        profile_score_pairs = sorted(profile_score_pairs, key = lambda x:x['rating_score'])
        return StopEvent(result=profile_score_pairs)
    
    @step(num_workers=4)
    async def profile_initialisation(self, ev: ProfileInitialisationEvent) -> ProfileGenerationEvent:
        positive_items = ev.positive_items
        dataset_context = ev.dataset_context
        # print('calling LLM...')
        rating1_items = [item for item in positive_items if item['user_rating'] == 1]
        rating2_items = [item for item in positive_items if item['user_rating'] == 2]
        rating3_items = [item for item in positive_items if item['user_rating'] == 3]
        rating4_items = [item for item in positive_items if item['user_rating'] == 4]
        rating5_items = [item for item in positive_items if item['user_rating'] == 5]

        message = PromptTemplate(self.prompt_dict['initialisation_prompt2']).format_messages(
            item_name = self.item_name,
            dataset_context = dataset_context,
            rating1_items = utils.format_postive_items(rating1_items),
            rating2_items = utils.format_postive_items(rating2_items),
            rating3_items = utils.format_postive_items(rating3_items),
            rating4_items = utils.format_postive_items(rating4_items),
            rating5_items = utils.format_postive_items(rating5_items),
        )
        try:
            raw_rating_based_response = utils.get_completion(self.client, message, self.config['model'], self.config['temperature'])
        except:
            try:
                raw_rating_based_response = utils.get_completion(self.client, message, self.config['model'], self.config['temperature'])
            except Exception as e:
                print(f"!!!!!!!ERR: {e}!!!!!!!")
                return ProfileGenerationEvent(initialised_profile = '', rating_score=999)
        
        initial_profile = raw_rating_based_response
        if initial_profile:
            initial_profile = utils.quote_process(initial_profile)
        else:
            initial_profile = ''
        
        # rating_score = await self.rating_evaluation(initial_profile, np.random.choice(positive_items, 10, replace=False))
        rating_score = 1
        return ProfileGenerationEvent(initialised_profile = initial_profile, rating_score = rating_score)



    async def rating_evaluation(self, profile, positive_items, iterater_round = 2):
        rmse = []
        random.shuffle(positive_items)
        for i in range(iterater_round):
            y_list = [item['user_rating'] for item in positive_items]
            message = PromptTemplate(self.prompt_dict['rating_prompt']).format_messages(
                item_name = self.item_name,
                user_profile = profile,
                positive_items = utils.recommendation_list_to_prompt(positive_items),
                samples_num = len(positive_items)
            )
            try:
                selection_response = await self.merge_llm.achat(messages = message)
            except Exception as e:
                print(f"Error during LLM call: {e}")
                continue
            # if selection_response
            # token_usage.append(selection_response.raw.usage)
            if selection_response.message.content:
                selection_response = utils.general_parser(selection_response.message.content)
            else:
                print(f"Rating Eval Error: {selection_response}")
                continue
            pred_list = self.get_rating_list(selection_response)
            if len(y_list) != len(pred_list):
                print(f"{y_list}\n{selection_response}")
            rmse.append(utils.calculate_rmse(y_list, pred_list))
            
        return sum(rmse)/len(rmse)
    
    def get_rating_list(self, llm_response):
        pattern = r"^\s*(.+?):\s+(\d+(?:\.\d+)?)/5"
        # re.MULTILINE ensures '^' matches the start of each line, not just the start of the string
        matches = re.findall(pattern, llm_response, re.MULTILINE)
        parsed_score = [float(score) for _, score in matches]
        
        return parsed_score



if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
