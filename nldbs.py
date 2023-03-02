import torch
import clip
import os

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from torchvision import transforms
import duckdb

import config_tdb
# from atr_example import AudioDataset, load_audio_model
from search import CraigslistDataset
from nlfilter import ImageProcessor, TextProcessor, AudioProcessor
from schema import NLDatabase, NLTable, NLColumn, DataType


models = {}


def get_text_model(device_id):
    if 'text' not in models:
        if config_tdb.USE_BART:
            models['text'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
        else:
            models['text'] = SentenceTransformer('all-MiniLM-L6-v2')
    return models['text']


def get_image_model(device):
    if 'image' not in models:
        models['image'] = clip.load('RN50', device)
    return models['image']


# def get_audio_model(device):
#     if 'audio' not in models:
#         models['audio'] = load_audio_model(device)
#     return models['audio']


def get_nldb_by_name(dbname):
    if dbname == 'craigslist':
        return craigslist()
    elif dbname == 'youtubeaudios':
        return youtubeaudios()
    else:
        raise ValueError(f'Wrong nldb name: {dbname}')


def read_csv_furniture():
    df = pd.read_csv('craigslist/furniture.tsv', sep='\t', index_col=0)
    df['aid'] = df.index
    df['title_u'] = df.index
    df.drop('imgs', axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time']).astype(np.int64) / 10 ** 9
    return df


def read_csv_youtube():
    df = pd.read_csv('audiocaps/youtube.csv')
    df['description'] = df['description'].fillna('')
    df['likes'] = df['likes'].fillna(0)
    df['description_u'] = np.arange(len(df))
    df['audio'] = np.arange(len(df))
    return df


def get_craiglist_images():
    # Load image dataset and processor for img column.
    img_dir = 'craigslist/furniture_imgs/'
    # img_dir = 'flickr/flickr30k_images/'
    img_paths = [img_dir + f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    t = transforms.Compose([
        transforms.ToPILImage()
    ])
    return CraigslistDataset(img_paths, t)


def craigslist():
    print(f'Initializing NL Database: Craigslist')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    model, preprocess = get_image_model(device)

    dataset = get_craiglist_images()
    processor = ImageProcessor(dataset, model, preprocess, device)

    # Read furniture table from csv.
    df = read_csv_furniture()
    # Initialize tables in duckdb.
    con = duckdb.connect(database=':memory:')  #, check_same_thread=False)
    # Register furniture table.
    con.execute("CREATE TABLE furniture AS SELECT * FROM df")
    con.execute("CREATE UNIQUE INDEX furniture_aid_idx ON furniture (aid)")
    print(f'len(furniture): {len(df)}')
    # Register image table.
    con.execute("CREATE TABLE images(img INTEGER PRIMARY KEY, aid INTEGER)")
    nr_imgs = len(dataset)
    con.executemany("INSERT INTO images VALUES (?, ?)",
                    [[idx, dataset[idx][1].split('_')[0]] for idx in range(nr_imgs)])
    print(f'len(images): {nr_imgs}')

    # Load text dataset and processor for the title column.
    text_model = get_text_model(device_id)
    title_processor = TextProcessor(df['title'], text_model, device)

    # Create NL database.
    furniture = NLTable('furniture')
    furniture.add(NLColumn('aid', DataType.NUM),
                  NLColumn('time', DataType.NUM),
                  NLColumn('neighborhood', DataType.TEXT),
                  NLColumn('title', DataType.TEXT),
                  NLColumn('title_u', DataType.NUM, title_processor),
                  NLColumn('url', DataType.TEXT),
                  NLColumn('price', DataType.NUM))
    images = NLTable('images')
    images.add(NLColumn('img', DataType.IMG, processor),
               NLColumn('aid', DataType.NUM))
    nldb = NLDatabase('craigslist', con)
    nldb.add(furniture, images)
    nldb.add_relationships(('furniture', 'aid', 'images', 'aid'))
    # Initialize metadata information.
    nldb.init_info()
    return nldb


def youtubeaudios():
    print(f'Initializing NL Database: YoutubeAudios')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    # Read youtube table.
    df = read_csv_youtube()
    # Load audio dataset and processor for the audio column. Only include valid audios.
    # dataset = AudioDataset(valid_idxs=df['audio'])
    # model = get_audio_model(device)
    # print(f'GPU - Audio Model: {next(model.parameters()).is_cuda}')
    # processor = AudioProcessor(dataset, model, device)
    # Register youtube table. Should be done after updating the audio column.
    con = duckdb.connect(database=':memory:')  #, check_same_thread=False)
    con.execute("CREATE TABLE youtube AS SELECT * FROM df")

    # Load text dataset and processor for the description column.
    text_model = get_text_model(device_id)
    description_processor = TextProcessor(df['description'], text_model, device)

    # Create NL database.
    youtube = NLTable('youtube')
    youtube.add(NLColumn('youtube_id', DataType.TEXT),
                # NLColumn('audio', DataType.AUDIO, processor),
                NLColumn('title', DataType.TEXT),
                NLColumn('category', DataType.TEXT),
                NLColumn('viewcount', DataType.NUM),
                NLColumn('author', DataType.TEXT),
                NLColumn('length', DataType.NUM),
                NLColumn('duration', DataType.TEXT),
                NLColumn('likes', DataType.NUM),
                NLColumn('description', DataType.TEXT),
                NLColumn('description_u', DataType.NUM, description_processor))
    nldb = NLDatabase('youtubeaudios', con)
    nldb.add(youtube)
    # Initialize metadata information.
    nldb.init_info()
    return nldb
