import pandas as pd
import json
import os
import argparse

from llm_generator_pipeline.config import load_config


# INPUT_COLUMNS = ['entity_id', 'name_en', 'name_ru', 'name_ar', 'name_zh', 'name_ja', 'name_he', 'name_hi', 'name_el', 'name_ko']

SCRIPT_COLUMNS = ['name_ru', 'name_ar', 'name_zh', 'name_ja', 'name_he', 'name_hi', 'name_el', 'name_ko']




def load_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found")
    data = pd.read_csv(dataset_path, dtype=str).fillna("")
    return data

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config.yaml", help="path to the config file")
        parser.add_argument("--input", type=str, default="data/names.csv", help="path to the names.csv")

        args = parser.parse_args()

        data = load_data(args.input)
        data["name_en"] = data["name_en"].astype(str).str.strip()
        data = data[data["name_en"] != ""].copy()
        # we always want to have name en so filter out the empty values
        data['bucket'] = data.apply(create_bucket, axis=1)

        config = load_config(args.config)
        sample_size = config["sample_size"]
        sample_seed = config["sample_seed"]

        stratified_df = stratified_sample(data, sample_size, sample_seed)
        write_to_jsonl(stratified_df)
        print(f"sucessfully created a stratified sample of size {len(stratified_df)} and saved it to data/pipeline/01_sampled.jsonl")
    except Exception as e:
        raise RuntimeError(f"Error in 01_sample.py: {e}")

def create_bucket(row):
    coverage_score = sum(bool(str(row[col]).strip()) for col in SCRIPT_COLUMNS)
    if coverage_score == 0:
        return "0"
    elif coverage_score <= 2:  
        return "1-2"
    elif coverage_score <= 4:
        return "3-4"
    else:
        return "5+"


# https://www.kaggle.com/code/flaviobossolan/stratified-sampling-python

def stratified_sample(df, size, seed):
    # number of rows in the dataset
    population = len(df)

    # if the requested sample size is larger than the population, return the entire dataset
    if size >= population:
        # frac 1 means return all rows, random_state seed makes it reporprducible, reset_index True resets row numbers 
        # reason => because then the fraction is 1 or bigger, meaning that you are basically asking for the whole dataset
        # we do not want to deal with bigger buckets which have more samples than the population, so we just return the whole dataset
        # basically, you would be asking for more samples than the population, which is not possible, so we just return the whole dataset
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # temporary dataframe which is used to count how many rows belong to each bucket
    # because you need to know the natural distrobution of the buckets
    tmp = df['bucket'].value_counts().reset_index()

    # renaming the columns for convenience purposes, before it was 'index' and 'bucket', now it is 'bucket' and 'size'
    tmp.columns = ['bucket', 'size']

    # calculating the proportional sample sizes, so in the case that for instance a bucket has 20 percent of the population, then it should have 20 percent of the sample size, so we are calculating the propotional sample size for each bucket
    # so basically, you take that number of rows from that bucket

    # simpler version:
    # tmp['samp_size'] = round(size/population * tmp['size']).astype(int)
    # more advanced version, which ensures that the total sample size is exactly equal to the request sample size, otherwise, because of rounding, it could be a bit smaller or bigger
    # solution => 
    # 1. calculate the sample size for each bucket
    # 2. calculate the difference between the requestd sample size and the acryual sample size
    # 3. sort the buckets, this is so that we can add the remaining samples to the buckets with the biggest fractional part (they would be affected more by the rounding problem)
    # 4. basically add the remaining samples to the buckets until the sample size is equal to thr requested sample size
    tmp['samp_size'] = (size/population * tmp['size']).astype(int) 
    tmp['fraction_part'] = (size/population * tmp['size']) - tmp['samp_size'] #basically, this is the fractional part that is lost due to rounding we do when we convert it to integer

    for i in range(size - tmp['samp_size'].sum()):
        # add the remaining samples to the buckets with the biggest fractional part
        tmp.loc[tmp['fraction_part'].idxmax(), 'samp_size'] += 1
        # after this, the fractional part of the bucket should become 0
        tmp.loc[tmp['fraction_part'].idxmax(), 'fraction_part'] = 0

    # storing the final stratified sample from each bucket; therefore, we can append them later on
    strat_sampled = []
    # looping over the buckets (0, 1-2, 3-4, 5+)
    for i in range(len(tmp)):
        # reading the bucket name
        value = tmp.iloc[i]['bucket']
        # number of rows to sample from that bucket
        n = tmp.iloc[i]['samp_size']
        # in the case that the bucket has less rows than the calculated sample size, then just take all the rows from that bucket
        # our if size >= population alreadt handles this issue, but just in case for instance if there is a rounding issue, then just to make sure we also take the minimum
        n = min(n, tmp.iloc[i]['size']) 
        # getting all rows that belong to that bucket
        full_df = df[df['bucket'] == value]

         
        # final dataframe
        if n > 0:
            tmp_df = full_df.sample(n=n, random_state=seed).reset_index(drop=True)
            strat_sampled.append(tmp_df)
    # combining all the stratified samples from each bucket into one
    stratified_df = pd.concat(strat_sampled, ignore_index=True)
    # shuffling the final stratified sample; otherwise, its ordered y bucket, making it random
    stratified_df = stratified_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
    return stratified_df


def write_to_jsonl(results):

# {
#   "entity_id": "Q12345",
#   "name_en": "Catherine",
#   "wikidata": {
#     "name_ru": "Екатерина",
#     "name_ar": "كاثرين",
#     "name_zh": "",
#     "name_ja": "",
#     "name_he": "",
#     "name_hi": "",
#     "name_el": "",
#     "name_ko": ""
#   }
    os.makedirs("data/pipeline", exist_ok=True)
    with open("data/pipeline/01_sampled.jsonl", "w", encoding="utf-8") as f:
        for _, row in results.iterrows():
            entry = {
                "entity_id": str(row['entity_id']).strip(),
                "name_en": str(row['name_en']).strip(),
                "wikidata": {
                    "name_ru": str(row['name_ru']).strip(),
                    "name_ar": str(row['name_ar']).strip(),
                    "name_zh": str(row['name_zh']).strip(),
                    "name_ja": str(row['name_ja']).strip(),
                    "name_he": str(row['name_he']).strip(),
                    "name_hi": str(row['name_hi']).strip(),
                    "name_el": str(row['name_el']).strip(),
                    "name_ko": str(row['name_ko']).strip()
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
		

if __name__ == "__main__":
	main()