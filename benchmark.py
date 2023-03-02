import itertools

import re
import string

import pandas as pd
import time
import datetime
import random

import config_tdb
import nldbs
from query_info import NLQuery



# leather sofa: 27.22 41/14674
# wooden chair: 23.66 466/14674
# dining table: 23.26 875/14674
# sofa: 19.60 1675/14674
# table: 20.57 2762/14674
# wooden: 16.02 9243/14674

def queries_craigslist():
    templates = [
        # "select {agg} from furniture where {f_pred}",
        # "select * from furniture where {f_pred} limit 10",
        "select * from images where {i_pred} limit 10",
        "select {agg} from images, furniture where {any_pred} and images.aid = furniture.aid",
        "select * from images, furniture where {any_pred} and images.aid = furniture.aid limit 10",
        "select {agg} from images, furniture where ({i_pred} and {f_pred}) and images.aid = furniture.aid",
        "select * from images, furniture where ({i_pred} and {f_pred}) and images.aid = furniture.aid limit 10",
        "select {agg} from images, furniture where ({i_pred} or {f_pred}) and images.aid = furniture.aid",
        "select * from images, furniture where ({i_pred} or {f_pred}) and images.aid = furniture.aid limit 10",
    ]
    col2conditions = {
        'img': ['blue chair'], # ['leather sofa', 'wooden chair', 'dining table', 'sofa', 'table', 'wooden'], # ['blue chair'],  # ['blue chair', 'table', 'wooden'],
        'title_u': ['wood']  # ['good condition', 'wood']
    }
    col2preds = {col: [f"nl({col}, '{condition}')" for condition in conditions] for col, conditions in col2conditions.items()}
    agg_funcs = ['min', 'max', 'sum', 'avg']
    agg_cols = ['price']  # ['price', 'time']
    aggs = ['count(*)'] + [f'{func}({col})' for func, col in itertools.product(agg_funcs, agg_cols)]
    type2preds = {
        'agg': aggs,
        'f_pred': col2preds['title_u'],
        'i_pred': col2preds['img'],
        'any_pred': col2preds['title_u'] + col2preds['img']
    }
    queries = []
    for template in templates:
        var_types = [v[1] for v in string.Formatter().parse(template) if v[1] is not None]
        for var_instances in itertools.product(*[type2preds[var_type] for var_type in var_types]):
            query = re.sub('\{.*?\}', '{}', template).format(*var_instances)
            print(query)
            queries.append(query)
    print(len(queries))
    return queries


def queries_youtubeaudios():
    templates = [
        "select {agg} from youtube where {any_pred}",
        "select * from youtube where {any_pred} limit 10",
        "select {agg} from youtube where {a_pred} and {d_pred}",
        "select * from youtube where {a_pred} and {d_pred} limit 10",
        "select {agg} from youtube where {a_pred} or {d_pred}",
        "select * from youtube where {a_pred} or {d_pred} limit 10",
    ]
    col2conditions = {
        'audio': ['voices'],
        'description_u': ['cooking']  # 'cooking', 'driving'
    }
    col2preds = {col: [f"nl({col}, '{condition}')" for condition in conditions] for col, conditions in col2conditions.items()}
    agg_funcs = ['min', 'max', 'sum', 'avg']
    agg_cols = ['likes']  # ['likes', 'length', 'viewcount']
    aggs = ['count(*)'] + [f'{func}({col})' for func, col in itertools.product(agg_funcs, agg_cols)]
    type2preds = {
        'agg': aggs,
        'a_pred': col2preds['audio'],
        'd_pred': col2preds['description_u'],
        'any_pred': col2preds['audio'] + col2preds['description_u']
    }
    queries = []
    for template in templates:
        var_types = [v[1] for v in string.Formatter().parse(template) if v[1] is not None]
        for var_instances in itertools.product(*[type2preds[var_type] for var_type in var_types]):
            query = re.sub('\{.*?\}', '{}', template).format(*var_instances)
            print(query)
            queries.append(query)
    print(len(queries))
    return queries

"""
Query: Ground Truth Threshold, Selectivity

leather sofa: 27.22 41/14674
blue chair: 21.63, 180/14674
desk: 21.71 396/14674
wooden chair: 23.66 466/14674
dining table: 23.26, 875/14674
chair: 22.33 925/14674
cozy sofa: 20.71, 1056/14674
sofa: 19.60 1675/14674
wood 17.94 2617/14674
table: 20.57 2762/14674
wooden: 16.02 9243/14674

delivery: 0.3162 53/3000
good condition: 0.3201 74/3000
new: 0.2131 199/3000
wood: 0.351764 657/3000
something more general: kitchen? office?

BART
wood: 0.9013 527/3000

music 0.465 18/46774
cooking 0.28863 793/46774
driving 0.2256 7989/46774

BART
cooking: cooking 0.8506 1117/46774

voices: -28.67618203163147 -28.675654530525208 49/3269 = ?/46774
"""
ground_truths = {'leather sofa': 27.22, 'desk': 21.71, 
                'wooden chair': 23.66, 'dining table': 23.26, 'chair': 22.33,
                'cozy sofa': 20.71, 'sofa': 19.60, 'table': 20.57, 'wooden': 16.02,  # craigslist
                    'blue chair': 21.63,  # craigslist
                    'good condition': 0.3201,  # craigslist
                    'wood': 0.9013 if config_tdb.USE_BART else 0.351764,  # craigslist
                    'cooking': 0.8506 if config_tdb.USE_BART else 0.28863,  # youtubeaudios
                    'voices': -32.8471, 'driving': 0.2256  # youtubeaudios
                    }


def main():
    constraints = [('error', 0.1, 1), ('error', 0.1, 10000), ('runtime', 10, 1), ('runtime', 10, 10000), ('feedback', 5, 1), ('feedback', 5, 10000)]  # ('runtime', 10, 10), ('feedback', 5, 100), ('error', 0.1, 10)  # (constrained metric, value, ratio versus runtime)
    methods = ['local']  # ['local', ('cost', 1), ('cost', 2), ('cost', 5), 'rl', 'random', 'ordered', 'naive']
    use_cache = False
    shuffle_queries = False
    random.seed(1)

    # sqls_craigslist = [
        # "select count(*) from images, furniture where nl(title_u, 'wood') and images.aid = furniture.aid",
        # "select * from furniture where nl(title_u, 'good condition') limit 10",
        # "select * from images where nl(img, 'wooden') limit 10",
        # "select min(price) from images, furniture where images.aid = furniture.aid and nl(img, 'wooden')",
        # "select max(price) from images, furniture where images.aid = furniture.aid and nl(img, 'wooden')",
        # "select * from images, furniture where (nl(img, 'wooden') and nl(title_u, 'good condition')) and images.aid = furniture.aid limit 10",
        # "select * from images, furniture where (nl(img, 'wooden') and nl(title_u, 'good condition')) and images.aid = furniture.aid limit 3",
        # "select * from images, furniture where (nl(img, 'wooden') or nl(title_u, 'good condition')) and images.aid = furniture.aid limit 10",
        # "select sum(price) from images, furniture where (nl(img, 'wooden') and nl(title_u, 'good condition')) and images.aid = furniture.aid",
        # "select avg(price) from images, furniture where (nl(img, 'wooden') and nl(title_u, 'good condition')) and images.aid = furniture.aid",
        # "select sum(price) from images, furniture where (nl(img, 'wooden') or nl(title_u, 'good condition')) and images.aid = furniture.aid",
        # "select sum(price) from images, furniture where images.aid = furniture.aid and nl(img, 'wooden')"
        # "select avg(price) from furniture where nl(title_u, 'wood')",
        # "select avg(price) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid",
        # "select avg(price) from images, furniture where (nl(img, 'blue chair') and nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select avg(price) from images, furniture where (nl(img, 'blue chair') or nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select sum(price) from furniture where nl(title_u, 'wood')",
        # "select sum(price) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid",
        # "select sum(price) from images, furniture where (nl(img, 'blue chair') and nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select sum(price) from images, furniture where (nl(img, 'blue chair') or nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select count(*) from furniture where nl(title_u, 'wood')",
        # "select count(*) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid",
        # "select count(*) from images, furniture where (nl(img, 'blue chair') and nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select count(*) from images, furniture where (nl(img, 'blue chair') or nl(title_u, 'wood')) and images.aid = furniture.aid",
        # "select min(price) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid",
        # "select max(price) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid",
        # "select * from images where nl(img, 'blue chair') limit 10",
        # "select max(time) from images, furniture where nl(img, 'blue chair') and images.aid = furniture.aid"
    # ]
    sqls_craigslist = queries_craigslist()
    if shuffle_queries:
        random.shuffle(sqls_craigslist)
        for sql in sqls_craigslist:
            print(sql)

    # TODO: Timeout for baselines.
    # sqls_youtubeaudios = [
    #     "select avg(likes) from youtube where nl(description_u, 'cooking')",
    #     "select avg(likes) from youtube where nl(audio, 'voices')",
    #     "select avg(likes) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select avg(likes) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')",
    #     "select sum(likes) from youtube where nl(description_u, 'cooking')",
    #     "select sum(likes) from youtube where nl(audio, 'voices')",
    #     "select sum(likes) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select sum(likes) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')",
    #     "select count(*) from youtube where nl(description_u, 'cooking')",
    #     "select count(*) from youtube where nl(audio, 'voices')",
    #     "select count(*) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select count(*) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')",
    #     "select max(likes) from youtube where nl(description_u, 'cooking')",
    #     "select * from youtube where nl(audio, 'voices') limit 10",
    #     "select min(likes) from youtube where nl(audio, 'voices')",
    #     "select min(length) from youtube where nl(audio, 'voices')",
    #     "select max(viewcount) from youtube where nl(audio, 'voices')",
    #     "select max(likes) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select min(likes) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select max(likes) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')",
    #     "select min(likes) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')",
    #     "select sum(viewcount) from youtube where nl(audio, 'voices') and nl(description_u, 'cooking')",
    #     "select sum(viewcount) from youtube where nl(audio, 'voices') or nl(description_u, 'cooking')"
    # ]
    sqls_youtubeaudios = queries_youtubeaudios()
    if shuffle_queries:
        random.shuffle(sqls_youtubeaudios)
        for sql in sqls_youtubeaudios:
            print(sql)

    dbname_to_sqls = {
        'craigslist': sqls_craigslist,
        'youtubeaudios': sqls_youtubeaudios
    }

    dict_csv = {'timestamp': [],
                'db': [],
                'method': [],
                'constraint': [],
                'sql': [],
                'total_time': [],
                'time_optimize': [],
                'time_sql': [],
                'time_ml': [],
                'time_rl': [],
                'time_sql_in_rl': [],
                'time_recreate_in_rl': [],
                'time_insert_in_rl': [],
                'time_query_in_rl': [],
                'error': [],
                'estimated_cost': [],
                'percent_images': [],
                'nr_feedbacks': [],
                'nr_actions': [],
                'cost': []}
    pd.DataFrame(dict_csv).to_csv('log/benchmark.csv', mode='a', index=False)
    for constraint in constraints:
        for dbname, sqls in dbname_to_sqls.items():
            for method in methods:
                has_init_nldb = False
                for sql in sqls:
                    nl_permutations = itertools.permutations(range(0, len(NLQuery(sql).nl_preds))) if method == 'ordered' else range(0, 1)
                    for nl_permutation in nl_permutations:
                        print(sql)
                        query = NLQuery(sql)
                        if not has_init_nldb or not use_cache:
                            cache = None
                            nldb = nldbs.get_nldb_by_name(dbname)
                            has_init_nldb = True
                        start = time.time()
                        if method == 'naive':
                            info = nldb.run_baseline(query, ground_truths)
                        elif method == 'ordered':
                            info = nldb.run_baseline_ordered(query, nl_permutation, ground_truths)
                        elif method == 'rl' or method == 'random' or method == 'local' or method[0] == 'cost':
                            info, cache = nldb.run(query, constraint, ground_truths, method, cache)
                        else:
                            raise ValueError(f'Unknown method name: {method}')
                        end = time.time()
                        print(f"RESULT {method} {constraint} {sql} - TIME/COST/IMAGES/FEEDBACKS: {end - start} {info['estimated_cost']} {info['processed']} {info['feedback']}")

                        dict_csv['timestamp'].append(str(datetime.datetime.now()))
                        dict_csv['db'].append(dbname)
                        dict_csv['method'].append(method)
                        dict_csv['constraint'].append(constraint)
                        dict_csv['sql'].append(sql)
                        dict_csv['total_time'].append(end - start)
                        dict_csv['time_optimize'].append(info.get('time_optimize'))
                        dict_csv['time_sql'].append(info.get('time_sql'))
                        dict_csv['time_ml'].append(info.get('time_ml'))
                        dict_csv['time_rl'].append(info.get('time_rl'))
                        dict_csv['time_sql_in_rl'].append(info.get('time_sql_in_rl'))
                        dict_csv['time_recreate_in_rl'].append(info.get('time_recreate_in_rl'))
                        dict_csv['time_insert_in_rl'].append(info.get('time_insert_in_rl'))
                        dict_csv['time_query_in_rl'].append(info.get('time_query_in_rl'))
                        dict_csv['error'].append(info['error'])
                        dict_csv['estimated_cost'].append(info['estimated_cost'])
                        dict_csv['percent_images'].append(info['processed'])
                        dict_csv['nr_feedbacks'].append(info['feedback'])
                        dict_csv['nr_actions'].append(info.get('nr_actions'))
                        if constraint[0] == 'error':
                            cost = (end - start) + constraint[2] * sum(info['feedback'])
                        elif constraint[0] == 'feedback':
                            cost = (end - start) + constraint[2] * info['error']
                        elif constraint[0] == 'runtime':
                            cost = sum(info['feedback']) + constraint[2] * info['error']
                        else:
                            raise ValueError(f'Unknown constraint option {constraint}')
                        dict_csv['cost'].append(cost)

                        pd.DataFrame(dict_csv).iloc[-1:].to_csv('log/benchmark.csv', mode='a', header=False, index=False)


if __name__ == "__main__":
    main()
