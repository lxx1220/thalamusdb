import math
import os
import pathlib
import random
import time

import pandas as pd
import numpy as np

import streamlit as st
import sys

import nldbs
from cost_optimizer import CostOptimizer
from datatype import DataType
from nlfilter import NLFilter
from query_info import NLQuery, NLQueryInfo


class DummyDB:
    """ Represents ThalamusDB implementation. """

    def __init__(self, dbname):
        """ Initialize for given database.


        Args:
            dbname: name of database to initialize for
        """
        if dbname == 'YouTube':
            dbname = 'youtubeaudios'
        elif dbname == 'Craigslist':
            dbname = 'craigslist'
        self.nldb = nldbs.get_nldb_by_name(dbname)
        self.query = None
        self.nl_filters = None
        self.fid2runtime = None

    def phase1(self, sql):
        """ Executes first phase of query processing: calculate
        similarity scores and identify objects to label.

        Args:
            query: an SQL query with natural language predicates
        """
        sql = sql.lower()
        self.query = NLQuery(sql)
        self.nl_filters = [NLFilter(self.nldb.get_col_by_name(col), text) for col, text in self.query.nl_preds]

        query = self.query
        nl_filters = self.nl_filters

        # Create scores tables.
        for fid, nl_filter in enumerate(nl_filters):
            self.nldb.con.execute(f"DROP TABLE IF EXISTS scores{fid}")
            self.nldb.con.execute(f"CREATE TABLE scores{fid}(sid INTEGER PRIMARY KEY, score FLOAT, processed BOOLEAN)")
            if nl_filter.idx_to_score:
                self.nldb.con.execute(
                    f"INSERT INTO scores{fid} VALUES {', '.join(f'({key}, {val}, TRUE)' for key, val in nl_filter.idx_to_score.items())}")
            # self.nldb.con.executemany(f"INSERT INTO scores{fid} VALUES (?, ?, ?)",
            #                      [[key, val, True] for key, val in nl_filter.idx_to_score.items()])
            # Add null scores for remaining rows.
            self.nldb.con.execute(f"""
                    INSERT INTO scores{fid} SELECT {nl_filter.col.name}, NULL, FALSE
                    FROM (SELECT {nl_filter.col.name}, score FROM {nl_filter.col.table} LEFT JOIN scores{fid} ON {nl_filter.col.table}.{nl_filter.col.name} = scores{fid}.sid) AS temp_scores
                    WHERE score IS NULL""")
        # Get all possible orderings.
        possible_orderings = [('uniform',)]
        for col_name in query.cols:
            possible_orderings.append(('min', col_name))
            possible_orderings.append(('max', col_name))
        # Preprocess some data.
        preprocess_percents = [nl_filter.default_process_percent for nl_filter in nl_filters]
        fid2runtime = []
        for fid, nl_filter in enumerate(nl_filters):
            start_nl_filter = time.time()
            # To prevent bias towards uniform sampling.
            percent_per_ordering = preprocess_percents[fid] / len(possible_orderings)
            for ordering in possible_orderings:
                action = ('i', 1, fid) if ordering == ('uniform',) else (
                    'o', 1, query.cols.index(ordering[1]), ordering[0], fid)
                self.nldb.process_unstructured(action, query, nl_filters, percent_per_ordering)
            end_nl_filter = time.time()
            time_nl_filter = end_nl_filter - start_nl_filter
            print(f'Unit process runtime: {time_nl_filter}')
            fid2runtime.append(time_nl_filter)
        self.fid2runtime = fid2runtime


@st.experimental_singleton
def load_db(database):
    print(f'Loading Database: {database}')
    db = DummyDB(database)
    print(f'Finished Loading Database: {database}')
    return db


@st.experimental_memo
def phase1_wrapper(_db, sql, optimizer_mode):
    # st.write(f'Processing phase 1 ...')
    db = _db
    print(f'Started Phase 1: {sql}')
    db.phase1(sql)
    st.session_state["label_idx"] = 0
    st.session_state["nr_extra_feedbacks"] = 0
    st.session_state["optimizer_mode"] = optimizer_mode  # 'local'
    st.session_state["computation_time"] = 0
    st.session_state["lus"] = []
    st.session_state["no_improvement"] = False
    print(f'Initialized Labeling Index: {st.session_state["label_idx"]}')
    print(f'Finished Phase 1: {sql}')


def labeling_request(nl_filter, fid):
    predicate = nl_filter.text
    item_info = nl_filter.streamlit_collect_user_feedback_get(0.5)
    if item_info is not None:
        target, idx = item_info
        st.write('Please label the following item:')
        if nl_filter.col.datatype == DataType.AUDIO:
            st.audio(target)
        elif nl_filter.col.datatype == DataType.IMG:
            st.image(target)
        else:
            st.markdown(f'{target}')
        label = st.radio(f'Predicate: {predicate}?', options=['Yes', 'No'], index=1, key=(nl_filter.text, fid, idx))

        if st.button('Submit Answer'):
            is_yes = label == 'Yes'
            nl_filter.streamlit_collect_user_feedback_put(is_yes, idx)
            st.session_state["label_idx"] += 1
            st.experimental_rerun()
    else:
        st.session_state["label_idx"] += 1
        st.experimental_rerun()


# def draw_bounds():
#     # Update bounds.
#     if st.session_state["lus"] is not None:
#         lus = st.session_state["lus"]
#         st.write('CURRENT QUERY RESULT:')
#         print('==========CURRENT QUERY RESULT==========')
#         chart_data = pd.DataFrame([['Lower Bound', lus[0][0]], ['Upper Bound', lus[0][1]]], columns=["labels", "bounds"])
#         chart_data = chart_data.set_index('labels')
#         print(chart_data)
#         st.bar_chart(chart_data)


def draw_line_bounds():
    # Update bounds.
    if st.session_state["lus"]:
        lus = st.session_state["lus"]
        st.write('Deterministic Bounds on Query Result:')
        print('==========CURRENT QUERY RESULT==========')
        # Upper should come first in order to display values when hovered.
        chart_data = pd.DataFrame(lus, columns=['Upper', 'Lower'])
        print(chart_data)
        st.area_chart(chart_data)


def process(db, sql, constraint, optimizer_mode):
    phase1_wrapper(db, sql, optimizer_mode)  # Cached.

    # st.write(f'Finished phase 1!')
    # st.write(f'Processing phase 2 ...')
    print(f'Current Labeling Index: {st.session_state["label_idx"]}')

    nl_filters = db.nl_filters
    # Collect 3 user feedbacks.
    preprocess_nr_feedbacks = 3
    total_preprocess_nr_feedbacks = len(nl_filters) * preprocess_nr_feedbacks
    total_nr_feedbacks = total_preprocess_nr_feedbacks + st.session_state["nr_extra_feedbacks"]
    if st.session_state["label_idx"] < total_nr_feedbacks:
        fid = st.session_state["fid"] if st.session_state["label_idx"] >= total_preprocess_nr_feedbacks \
            else (st.session_state["label_idx"] // preprocess_nr_feedbacks)
        nl_filter = nl_filters[fid]
        draw_line_bounds()
        labeling_request(nl_filter, fid)
    else:
        optimizer_mode = st.session_state["optimizer_mode"]
        # Run query
        action_queue = st.session_state["action_queue"] if "action_queue" in st.session_state else []
        is_error_constraint, is_feedback_constraint, is_runtime_constraint, error_constraint, feedback_constraint, runtime_constraint, metric_ratio = CostOptimizer.gather_constraint_info(
            constraint, len(nl_filters))
        no_improvement = st.session_state["no_improvement"]
        # while True:
        start_time = time.time()
        error, lus = db.nldb.query_to_compute_error(db.query, db.nl_filters, print_error=True, return_results=True)
        if not math.isinf(lus[0][0]) and not math.isinf(lus[0][1]):
            st.session_state["lus"].append((lus[0][1], lus[0][0]))
        cur_nr_feedbacks = st.session_state["label_idx"]
        st.session_state["computation_time"] += (time.time() - start_time)
        cur_runtime = st.session_state["computation_time"]
        draw_line_bounds()

        if not (is_error_constraint and error <= error_constraint) and \
                ((optimizer_mode == 'local' and not no_improvement)
                 or (optimizer_mode != 'local' and is_error_constraint and error > error_constraint)):
            # If queue is empty, add more actions.
            start_time = time.time()
            if not action_queue:
                if optimizer_mode == 'local':
                    query_info = NLQueryInfo(db.query)
                    optimizer = CostOptimizer(db.nldb, db.fid2runtime, metric_ratio)
                    max_nr_total = max([db.nldb.info.tables[name].nr_rows for name in db.query.tables])
                    actions = optimizer.optimize_local_search(query_info, nl_filters, max_nr_total, constraint, cur_nr_feedbacks, cur_runtime)
                    if not actions:
                        st.session_state["no_improvement"] = True
                    action_queue.extend(actions)
                elif optimizer_mode == 'random':
                    query_info = NLQueryInfo(db.query)
                    optimizer = CostOptimizer(db.nldb, db.fid2runtime, metric_ratio)
                    possible_actions = optimizer.get_all_possible_actions(len(nl_filters), len(query_info.query.cols))
                    print('Randomly selecting next action.')
                    action = random.choice(possible_actions)
                    action_queue.append(action)
                elif optimizer_mode[0] == 'cost':
                    # TODO: One-shot, Multi-shots (i.e., thinking multiple steps ahead), Step-by-step
                    look_ahead = optimizer_mode[1]
                    query_info = NLQueryInfo(db.query)
                    optimizer = CostOptimizer(db.nldb, db.fid2runtime, metric_ratio)
                    max_nr_total = max([db.nldb.info.tables[name].nr_rows for name in db.query.tables])
                    actions = optimizer.optimize(query_info, nl_filters, max_nr_total, look_ahead, constraint, cur_nr_feedbacks, cur_runtime)
                    action_queue.extend(actions)
                else:
                    raise ValueError(f'No such optimizer mode: {optimizer_mode}')
            st.session_state["computation_time"] += (time.time() - start_time)

            if action_queue:
                action = action_queue.pop(0)
                print(f'Next action: {action}')
                action_type = action[0]
                fid = action[-1]
                nl_filter = nl_filters[fid]
                if action_type == 'i' or action_type == 'o':  # or action_type == 'c':
                    start_time = time.time()
                    process_percent_multiple = action[1]
                    processed_percent = process_percent_multiple * nl_filter.default_process_percent
                    db.nldb.process_unstructured(action, db.query, nl_filters, processed_percent)
                    st.session_state["computation_time"] += (time.time() - start_time)
                elif action_type == 'u':
                    # user_feedback_opt = action[1]
                    st.session_state["nr_extra_feedbacks"] += 1
                    # st.session_state["action_queue"] = action_queue
                    st.session_state["fid"] = fid
                    # st.experimental_rerun()
                else:
                    raise ValueError(f'Our action is out of scope: {action}')

            st.session_state["action_queue"] = action_queue
            st.experimental_rerun()
        # When error below the given constraint.
        else:
            # Resort to greedy method if error constraint not satisfied.
            if optimizer_mode == 'local' and is_error_constraint and error > error_constraint:
                print(f'Resort to greedy method due to error constraint not satisfied: {error}')
                optimizer_mode = ('cost', 1)
                st.session_state["optimizer_mode"] = optimizer_mode
                st.experimental_rerun()

            st.write("Finished Processing!")
                # continue
                # Stop if satisfies the constraint.
                # st.write(f'Finished phase 2!')
                # print('==========QUERY RESULT==========')
                # error, lus = db.nldb.query_to_compute_error(db.query, nl_filters, print_error=True, print_result=True, return_results=True)
                #
                # chart_data = pd.DataFrame([['Lower Bound', lus[0][0]], ['Upper Bound', lus[0][1]]], columns=["labels", "bounds"])
                # chart_data = chart_data.set_index('labels')
                # print(chart_data)
                #
                # st.bar_chart(chart_data)

                # break


# Interface.
cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent
root_dir = src_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(root_dir))
# print(f'sys.path: {sys.path}')
print('RELOADING')

os.environ['KMP_DUPLICATE_LIB_OK']='True'
st.set_page_config(page_title='ThalamusDB')
st.markdown('''
# ThalamusDB
ThalamusDB answers complex queries with natural 
language predicates on multi-modal data.
''')

# SELECT Max(Price) FROM Images, Furniture WHERE NL(Img, 'wooden table') AND NL(Title_u, 'good condition') AND Images.Aid = Furniture.Aid

# TODO:
#  [O] Which tables and columns are in the database. Maybe a popup?
#  [O] Online processing: Show shrinking deterministic bounds.
#  [NA] One bar for bounds. Maybe x-axis instead of y-axis.
#  Some statistics/additional internal information that would be useful for the demo
#  - Streamlit expander. Selected query plan (estimated cost metrics).
#  [O] Randomized plan as baseline.
#  Symphony: Towards Natural Language Query Answering over Multi-modal Data Lakes.
#  Breakdown of processing time (baselines as well).
#  DDL statements to add new data (parse using sqlglot). Bulk loading from csv.

with st.sidebar:
    metrics = ['Error', 'Computation', 'Interactions']
    constraint_on = st.selectbox('Constrained Metric:', options=metrics)
    constraint_max = 1.0 if constraint_on == 'Error' else 1000.0
    constraint_value = float(
        st.slider(
            f'Upper Bound ({constraint_on}):',
            min_value=0.0, max_value=constraint_max, value=0.1))

    weights = [-1] * 3
    for metric_idx, metric in enumerate(metrics):
        if not constraint_on == metric:
            weights[metric_idx] = st.slider(
                f'Weight for {metric}:',
                min_value=1.0, max_value=1000.0)

database = st.selectbox('Database:', options=['Craigslist', 'YouTube'])
db = load_db(database)
with st.expander('Database Schema Info'): # expanded=True
    for table_name, table_info in db.nldb.info.tables.items():
        st.write(f'{table_name}'
                 f' ({", ".join(col_name for col_name in table_info.cols.keys() if not col_name.endswith("_u"))})')
sql = st.text_area('SQL Query with Natural Language Predicates:')

optimizer_mode = st.selectbox('Query Planning:', options=['Optimized', 'Non-optimized'])
optimizer_mode = 'local' if optimizer_mode == 'Optimized' else 'random'

if "process_input" not in st.session_state:
    st.session_state["process_input"] = None

if st.button("Process Query") or st.session_state["process_input"] == (database, sql, constraint_value, weights, optimizer_mode):
    st.session_state["process_input"] = (database, sql, constraint_value, weights, optimizer_mode)
    weight = None
    if constraint_on == 'Error':
        weight = weights[1] / weights[2]
    elif constraint_on == 'Computation':
        weight = weights[2] / weights[0]
    elif constraint_on == 'Interactions':
        weight = weights[1] / weights[0]
    else:
        raise NotImplementedError()
    constraint = (constraint_on.lower(), constraint_value, weight)
    process(db, sql, constraint, optimizer_mode)








