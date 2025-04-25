import os,json, string, csv, re
import zstandard
from collections import defaultdict
from datetime import datetime
import logging.handlers
import ast

import pandas as pd

#supervised ml tf-idt/lr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# import ipywidgets as widgets #I think these 2 libraries aren't used anymore
# from IPython.display import display, clear_output

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords') # just need this the first time
stopwords = set(stopwords.words('english'))

### SET UP VARIABLES ###
# skip everything before this date. 
start_date = datetime.strptime("2020-01-01", '%Y-%m-%d') 

from keyword_lists import filtering_phrases as phrases
patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in phrases] # precompile regex pattern

parent_directory = "big3zst" 
#parent_directory = "../reddit/subreddits23/medical_subreddits"

### STEP 1: ROUGH KEYWORD FILTERING ###
# Goal: do an initial rough cut to get text we think might be about cannabis + health

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0): #helper function
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name): #helper function
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()

#keyword search through zst files
#produces csv files of all the hits, and line_counts.csv that gives the prevalence of keyword hits
def zst_keyword_filter():  
    count_file = open("line_counts.csv", 'w')
    count_writer = csv.writer(count_file)
    count_writer.writerow(["file", "total_lines", "keyword_lines"])

    output_folder = "keyword_hits"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    directory = os.fsencode(parent_directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        input_path = parent_directory + "/" + filename # the path to the input comment file

        # Write output file
        output_name = None #set to null to catch any errors in below regex
        match = re.match(r'^([^_]+_[a-zA-Z])', filename) #subreddit name + _c or _s (comments, submissions)
        if match:
            shortname = match.group(1) 
            output_name = shortname + "_keyword_hits.csv" 
        else:
            print("Error in regex: no match found: " + filename) #this shouldn't ever happen 
            
        output_path = output_folder + "/" + output_name

        csv_file = open(output_path, 'w')
        writer = csv.writer(csv_file)
        writer.writerow(["id","text","keywords"])

        # bunch of initialization stuff
        file_lines = 0
        hit_lines = 0
        file_bytes_processed = 0
        created = None
        bad_lines = 0
        current_day = None
        input_size = os.stat(input_path).st_size

        try:
            # this is the main loop where we iterate over every single line in the zst file
            for line, file_bytes_processed in read_lines_zst(input_path):
                try:
                    obj = json.loads(line) # load the line into a json object
                    created = datetime.utcfromtimestamp(int(obj['created_utc'])) #date obj transform
                    
                    if created >= start_date: # skip if we're before the start date defined above
                        if "comments" in filename:
                            body_lower = obj['body'].lower() #probably redundant
                            if any(pattern.search(body_lower) for pattern in patterns): # checking for a match
                                matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(body_lower)]
                                writer.writerow([obj['id'], body_lower, matches])
                                hit_lines += 1
                        elif "submissions" in filename:
                            title_lower = obj['title'].lower()
                            text_lower = obj['selftext'].lower()
                            if text_lower == "[deleted]" or text_lower == "[removed]": #skip any posts with a missing body
                                continue
                            if any(pattern.search(title_lower) for pattern in patterns): 
                                matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(title_lower)]
                                writer.writerow([obj['id'], title_lower + " /// " + text_lower, matches]) #/// is just to separate the title & body text
                                hit_lines += 1
                            elif any(pattern.search(text_lower) for pattern in patterns):
                                matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(text_lower)]
                                writer.writerow([obj['id'], title_lower + " /// " + text_lower, matches])
                                hit_lines += 1
                            
                # just in case there's corruption somewhere in the file
                except (KeyError, json.JSONDecodeError) as err:
                    bad_lines += 1
                file_lines += 1
                if file_lines % 100000 == 0:
                    log.info(f" {filename} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / input_size) * 100:.0f}%")
        except Exception as err:
            log.info(err)

        log.info(f"{filename} Complete : {file_lines:,} : {bad_lines:,}")
            # the path to the output csv file of word counts

        #add total lines to a csv 
        count_writer.writerow([filename, file_lines, hit_lines])

        csv_file.close()
    count_file.close()


def zst_dosages_filter():  
    dosage_pattern = re.compile(r"\b\d+(\.\d+)?\s?(mg|g|ml|%)\b")
    from keyword_lists import compounds_short as phrases
    patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in phrases]

    count_file = open("line_counts_dosage.csv", 'w')
    count_writer = csv.writer(count_file)
    count_writer.writerow(["file", "total_lines", "keyword_lines"])

    output_folder = "keyword_hits"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    directory = os.fsencode(parent_directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        input_path = parent_directory + "/" + filename # the path to the input comment file

        # Write output file
        output_name = None #set to null to catch any errors in below regex
        match = re.match(r'^([^_]+_[a-zA-Z])', filename) #subreddit name + _c or _s (comments, submissions)
        if match:
            shortname = match.group(1) 
            output_name = shortname + "_keyword_hits.csv" 
        else:
            print("Error in regex: no match found: " + filename) #this shouldn't ever happen 
            
        output_path = output_folder + "/" + output_name

        csv_file = open(output_path, 'w')
        writer = csv.writer(csv_file)
        writer.writerow(["id","text","keywords"])

        # bunch of initialization stuff
        file_lines = 0
        hit_lines = 0
        file_bytes_processed = 0
        created = None
        bad_lines = 0
        current_day = None
        input_size = os.stat(input_path).st_size

        try:
            # this is the main loop where we iterate over every single line in the zst file
            for line, file_bytes_processed in read_lines_zst(input_path):
                try:
                    obj = json.loads(line) # load the line into a json object
                    created = datetime.utcfromtimestamp(int(obj['created_utc'])) #date obj transform
                    
                    if created >= start_date: # skip if we're before the start date defined above
                        if "comments" in filename:
                            body_lower = obj['body'].lower() #probably redundant
                            dosages = dosage_pattern.findall(body_lower)
                            if len(dosages) > 0:
                                if any(pattern.search(body_lower) for pattern in patterns): # checking for a match
                                    matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(body_lower)]
                                    writer.writerow([obj['id'], body_lower, matches])
                                    hit_lines += 1
                        elif "submissions" in filename:
                            title_lower = obj['title'].lower()
                            text_lower = obj['selftext'].lower()
                            if text_lower == "[deleted]" or text_lower == "[removed]": #skip any posts with a missing body
                                continue
                            dosages_title = dosage_pattern.findall(title_lower)
                            dosages_text = dosage_pattern.findall(text_lower)
                            if len(dosages_title) > 0 or len(dosages_text) > 0:
                                if any(pattern.search(title_lower) for pattern in patterns): 
                                    matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(title_lower)]
                                    writer.writerow([obj['id'], title_lower + " /// " + text_lower, matches]) #/// is just to separate the title & body text
                                    hit_lines += 1
                                elif any(pattern.search(text_lower) for pattern in patterns):
                                    matches = [kw for kw, pattern in zip(phrases, patterns) if pattern.search(text_lower)]
                                    writer.writerow([obj['id'], title_lower + " /// " + text_lower, matches])
                                    hit_lines += 1
                            
                # just in case there's corruption somewhere in the file
                except (KeyError, json.JSONDecodeError) as err:
                    bad_lines += 1
                file_lines += 1
                if file_lines % 100000 == 0:
                    log.info(f" {filename} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / input_size) * 100:.0f}%")
        except Exception as err:
            log.info(err)

        log.info(f"{filename} Complete : {file_lines:,} : {bad_lines:,}")
            # the path to the output csv file of word counts

        #add total lines to a csv 
        count_writer.writerow([filename, file_lines, hit_lines])

        csv_file.close()
    count_file.close()


#get frequencies of all the words from list phrases in the dataset, wordcounts.csv
#just some descriptives
def word_counts(csv_folder):  
    overall_keyword_freq = {}
    csv_file = open('wordcounts.csv', 'w')
    writer = csv.writer(csv_file)
    writer.writerow(["source", "keyword", "freq"])

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            this_keyword_freq = {}
            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    for keyword in ast.literal_eval(row['keywords']):
                        overall_keyword_freq[keyword] = overall_keyword_freq.get(keyword, 0) + 1
                        this_keyword_freq[keyword] = this_keyword_freq.get(keyword, 0) + 1
                for key, value in this_keyword_freq.items():
                    writer.writerow([filename, key, value])
            except:
                print("Error: " + filename)

    for key, value in overall_keyword_freq.items():
       writer.writerow(["overall", key, value])

    csv_file.close()

## TODOs for future iterations of this rough cut: 
## grab context of the comments and posts so you get the whole conversation 

### STEP 2: CLASSIFICATION ###
# Goal: refining the rough cut to a dataset of text about health/therapeutic effects of cannabis

### Option 2a: Manual labeling for ground truths
#take a random sample n from each file for training labels
def sample_rows(n, csv_folder): 
    sampled_rows = []

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            try:
                df = pd.read_csv(filepath)
                sample = df.sample(n=n, random_state=42) if len(df) >= n else df
                sample['source_file'] = filename  # track origin
                sampled_rows.append(sample)
            except:
                print("Error: " + filename)

    #combine
    sampled_df = pd.concat(sampled_rows, ignore_index=True)

    #this is getting crazy, leave it alone for now. trying to make sure there's a minimum of 10 of each keyword sampled
    # keyword_freq = {}
    # for _, row in sampled_df.iterrows():
    #     for keyword in ast.literal_eval(row['keywords']):
    #         keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1

    # for k in phrases:
    #     if keyword_freq.get(k, 0) < 10:
    #         for filename in os.listdir(csv_folder):
    #             if filename.endswith(".csv"):
    #                 filepath = os.path.join(csv_folder, filename)
    #                 df = pd.read_csv(filepath)
    #                 try:
    #                     moresamp = df.loc[df['keywords'].str.contains(k, case=False, na=False)].sample(n=10, random_state=42)
    #                 except:
    #                     moresamp = df.loc[df['keywords'].str.contains(k, case=False, na=False)]
    #                 moresamp['source_file'] = filename  # track origin
    #                 sampled_rows.append(moresamp)

    # #combine again
    # sampled_df = pd.concat(sampled_rows, ignore_index=True)
    # print(sampled_df)
    sampled_df.to_csv("sampled_dataset.csv", index=False)


#calculate the precision (true positives/all positives) of all keywords
#and create a new sample of keywords that were too rare to be selected much
#note that there's some hardcoding as I was just exploring the data
def false_positives(filename): 
    df = pd.read_csv(filename)

    flattened_rows = []
    for _, row in df.iterrows():
        for keyword in ast.literal_eval(row['keywords']):
            flattened_rows.append({'keyword': keyword, 'label': row['chanda_topic_classification']})

    flattened_df = pd.DataFrame(flattened_rows)
    keyword_counts = flattened_df.groupby(['keyword', 'label']).size().unstack(fill_value=0)
    keyword_counts.to_csv("false_positives.csv")

    sampled_keywords = [d['keyword'] for d in flattened_rows] #all the keywords that were included in the labeling sample
    unsampled_keywords = list(set(phrases) - set(sampled_keywords)) #keywords that were not selected and cannot be evaluated
    #possible todo, add another round of sampling here or change original sample algo 
    #so that all words have at least 10(?) examples in the sample on top of the random sample
    # print(unsampled_keywords)

    #hardcoded hack job version: redo the above with just these keywords and do another sample
    #this is NOT VALID FOR ACTUAL ANALYSIS but is creating a sample from just these rarer words
    insufficient_sample = ['delta-9', 'THCV', 'Mary Jane', 'HHC', 'broad spectrum', 'rosin', 'hashish', 
                            'distillate', 'ganja', 'live resin', 'diamonds', 'sticky icky', 'pre-roll', 
                            'dank', 'nug', 'CBG', 'Rick Simpson Oil', 'dope', 'kush', 'reefer', 'shatter', 
                            'RSO', 'crumble', 'delta 9', 'delta-8', 'hash', 'terpenes', 'herb', 'delta 8', 
                            'full spectrum', 'grass', 'tincture', 'CBN', 'sauce', 'hemp', 'wax', 'flower']

    csv_folder = "keyword_hits"  
    rows = 0
    sampled_rows = []
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            df = pd.read_csv(filepath)
            if rows >= 370:
                break
            for _, row in df.iterrows():
                lst = ast.literal_eval(row['keywords'])
                if len(set(insufficient_sample).intersection(set(lst))) > 0:
                    row['source_file'] = filename  # track origin
                    sampled_rows.append(pd.DataFrame([row]))
                    rows += 1
    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    sampled_df.to_csv("rare_sampled_dataset.csv", index=False)
 

## Option 2b: Supervised machine learning tf-idf+lr
def supervised_learning(labelfile):
    df = pd.read_csv(labelfile)
    df = df.dropna(subset=['chanda_topic_classification']) #redundancy to drop NAs
    df['label'] = df['chanda_topic_classification'].astype(int) 

    #split into test/train
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    #create & train pipeline 
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words='english', max_features=3000),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)

    #evaluate performance
    y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))


    #plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Relevant", "Relevant"]).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    #Feature Importance
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['logisticregression']

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'weight': coefficients
    }).sort_values(by='weight', ascending=False)

    print("\n--- Top 20 Positive Predictors ---")
    print(importance_df.head(20))

    print("\n--- Top 20 Negative Predictors ---")
    print(importance_df.tail(20))

    #Active Learning -  select most uncertain examples & write to csv
    from sklearn.metrics import pairwise_distances

    csv_folder = "keyword_hits/csvs_2"
    unlabeled_df = pd.concat([pd.read_csv(os.path.join(csv_folder, file)) for file in os.listdir(csv_folder)], ignore_index=True)
    unlabeled_df = unlabeled_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    #remove labeled rows - this probably isn't necessary on a practical level if it slows performance too much
    known_texts = set(df['text']) 
    unlabeled_df = unlabeled_df[~unlabeled_df['text'].isin(known_texts)].reset_index(drop=True)

    vectorizer = model.named_steps['tfidfvectorizer']
    X_pool = vectorizer.transform(unlabeled_df['text'])
    X_train_vec = vectorizer.transform(X_train)

    probs = model.named_steps['logisticregression'].predict_proba(X_pool)
    uncertainty = 1 - probs.max(axis=1)
    unlabeled_df['uncertainty'] = uncertainty

    uncertain_batch = unlabeled_df.sort_values(by='uncertainty', ascending=False).head(100)
    uncertain_batch.to_csv("active_learning_batch.csv", index=False)
    print("\nSaved 100 most uncertain examples to active_learning_batch.csv")
    return model

def supervised_learning_full(model):
    csv_folder = "keyword_hits/csvs_2"
    unlabeled_df = pd.concat([pd.read_csv(os.path.join(csv_folder, file)) for file in os.listdir(csv_folder)], ignore_index=True)
    unlabeled_df = unlabeled_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    unlabeled_df['predicted_label'] = model.predict(unlabeled_df['text'])
    unlabeled_df.to_csv("full_supervised_classification.csv", index=False)

def semantic_embeddings(labelfile):
    df = pd.read_csv(labelfile)
    df = df.dropna(subset=['chanda_topic_classification']) #redundancy to drop NAs
    df['label'] = df['chanda_topic_classification'].astype(int) 

    #split into test/train
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['chanda_topic_classification'], test_size=0.2, random_state=42
    )

    #convert text to embeddings
    X_train_embeddings = embedder.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embeddings = embedder.encode(X_test.tolist(), show_progress_bar=True)

    #train classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_embeddings, y_train)
    y_pred = classifier.predict(X_test_embeddings)

    #evaluate model
    print("\n--- Semantic Embedding Model Performance ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Relevant", "Relevant"]).plot(cmap="Purples")
    plt.title("Confusion Matrix (Embeddings Model)")
    plt.show()


def semantic_embeddings_full(model):
    csv_folder = "keyword_hits/csvs_2"
    unlabeled_df = pd.concat([pd.read_csv(os.path.join(csv_folder, file)) for file in os.listdir(csv_folder)], ignore_index=True)
    unlabeled_df = unlabeled_df.drop_duplicates(subset=['text']).reset_index(drop=True)

    unlabeled_texts = unlabeled_df['text'].tolist()
    unlabeled_embeddings = embedder.encode(unlabeled_texts, show_progress_bar=True)
    unlabeled_predictions = classifier.predict(unlabeled_embeddings)

    unlabeled_df['predicted_label'] = unlabeled_predictions
    unlabeled_df.to_csv("semantic_embeddings_classified.csv", index=False)

#possibly deprecated
def create_matrices():
    ## CREATING KEYWORD MATRICES ##
    parent_directory = "keyword_hits/" 
    directory = os.fsencode(parent_directory)

    for file in os.listdir(directory): #iterate through keyword filtered csv's
        filename = os.fsdecode(file)
        input_path = parent_directory + filename 
        try: 
            df = pd.read_csv(input_path)

            results = []
            for i, row_data in df.iterrows(): #get each existing field and append it to the row
                row = {
                    'id': row_data['id'],
                    'text': row_data['text'],
                    'keywords': row_data['keywords'] 
                    }
                text_lower = str(row_data['text']).lower()
                for phrase in phrases: # add the matrix values
                    row[phrase] = int(phrase in text_lower)
                results.append(row)

            results_df = pd.DataFrame(results) #turn into df

            # Write output file
            output_name = None #set to null to catch any errors in below regex
            match = re.match(r'^([^_]+_[a-zA-Z])', filename) #subreddit name + _c or _s (comments, submissions)
            if match:
                shortname = match.group(1) 
                output_name = shortname + "_matrix.csv" 
            else:
                print("Error in regex: no match found in " + filename)

            output_path = os.path.join(parent_directory, output_name)
            results_df.to_csv(output_path, index=False)

            print(f"Processed: {filename} → {output_name}")

        except:
            print("File skipped: " + input_path)



if __name__ == "__main__":
    ## Step 1: Rough keyword filtering - go from zsts to filtered csvs
    # zst_keyword_filter() #creates csv's of all the keyword hits + prevalence counts
    # word_counts("keyword_hits/csvs_2") #get frequencies of all the keywords
    # zst_dosages_filter() #looks for cannabis dosages only
    
    ## Step 2: Classification (final text selection)
    ## Option 2a: Manual classification for ground truths
    sample_rows(50, "keyword_hits") #produces (psuedo)random samples 
    # false_positives("sampled_dataset_LABELLED.csv") #calculates precision by keyword & samples some rare keyword hits for further labeling

    ## Option 2b: Keyword filtering
    ## see gpt-keyword-classification.py for some beginning code for this, skipping for now

    ## Option 2c: Supervised machine learning (TF-IDF + LR)
    #trains the model and will produce csvs of additional data to label for training each time it's run
    # model = supervised_learning("sampled_dataset_LABELLED.csv") 
    # supervised_learning_full(model) # for when the model is trained well enough

    ## Option 2d: ML semantic classification
    # model = semantic_embeddings("sampled_dataset_LABELLED.csv")
    # semantic_embeddings_full(model)

    ## Option 2e: GPT semantic classification
    # skipping for now bc I don't want to set up an API account






