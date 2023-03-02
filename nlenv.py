import math
import numpy as np
import time
import random
import gym
from gym import spaces

from datatype import DataType

runtimes = {'recreate': 0, 'insert': 0, 'query': 0,
'insert_sids': 0, 'insert_scores': 0, 'insert_update': 0}


def rl_action_to_cost(rl_action):
    return 1


class NLEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent should learn to choose the optimal action for the current environment.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    best = {'env': None, 'improvement': float('-inf')}

    def __init__(self, nldb, query, error_constraint, nl_filters, max_nr_actions=5):
        super(NLEnv, self).__init__()
        self.process_percent_multiple = 1
        self.nr_user_feedback_opts = 1

        self.nldb = nldb
        self.query = query
        self.error_constraint = error_constraint
        self.nl_filters = nl_filters
        self.max_nr_actions = max_nr_actions
        # self.con = con
        # self.nl_filter = nl_filter

        # Define action and observation space
        # They must be gym.spaces objects

        # Extract action space. User feedback + Image Processing.
        self.nr_actions_per_filter = self.nr_user_feedback_opts + 1 + 2 * len(self.query.cols)
        nr_actions = self.nr_actions_per_filter * len(nl_filters)
        self.action_space = spaces.Discrete(nr_actions)

        # Extract observation space (i.e., node).
        # Set of actions.
        # 1. the first object contains the lower and upper thresholds.
        # 2. the second object contains the percentage of images we processed.
        # self.observation_space = spaces.Dict(
        #     {
        #         "thresholds": spaces.Box(low=-temp_inf, high=temp_inf, shape=(2,), dtype=float),
        #         "processed": spaces.Box(low=0, high=100, shape=(1,), dtype=float),
        #     }
        # )
        lowers = []
        uppers = []
        self.start_node = []
        for nl_filter in self.nl_filters:
            nr_processed = len(nl_filter.idx_to_score)
            lowers.extend([nl_filter.lower, nl_filter.lower, 0])
            uppers.extend([nl_filter.upper, nl_filter.upper, nl_filter.col.processor.nr_total])
            self.start_node.extend([nl_filter.lower, nl_filter.upper, nr_processed])
        self.nr_node_vals_per_filter, remainder = divmod(len(lowers), len(nl_filters))
        assert remainder == 0
        self.observation_space = spaces.Box(low=np.array(lowers), high=np.array(uppers), shape=(len(lowers),), dtype=float)

        # Initialize starting position.
        self.node = np.array(self.start_node, dtype=float)
        self.cur_step = 0
        start_recreate = time.time()
        self.nldb.recreate_scores_table(self.nl_filters)
        end_recreate = time.time()
        global runtimes
        runtimes['recreate'] += end_recreate - start_recreate
        start_query = time.time()
        self.prev_error = self.nldb.query_to_compute_error(self.query, self.nl_filters)  # Needs to be after re-creating the scores table
        end_query = time.time()
        # global runtimes
        runtimes['query'] += end_query - start_query
        # self.first_action = None

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent.
        self.node = np.array(self.start_node, dtype=float)
        self.cur_step = 0
        start_recreate = time.time()
        self.nldb.recreate_scores_table(self.nl_filters)
        end_recreate = time.time()
        global runtimes
        runtimes['recreate'] += end_recreate - start_recreate
        start_query = time.time()
        self.prev_error = self.nldb.query_to_compute_error(self.query, self.nl_filters)  # Needs to be after re-creating the scores table
        end_query = time.time()
        # global runtimes
        runtimes['query'] += end_query - start_query
        # self.first_action = None
        return self.node

    def step(self, action):
        self.cur_step += 1

        # Update state.
        option = self.action_to_option(action)
        fid = option[-1]
        nl_filter = self.nl_filters[fid]
        cost = rl_action_to_cost(action)
        if option[0] == 'i' or option[0] == 'o':
            process_percent_multiple = option[1]
            processed_percent = process_percent_multiple * nl_filter.default_process_percent

            node_idx = (1 + fid) * self.nr_node_vals_per_filter - 1
            nr_to_process = min(int(processed_percent * nl_filter.col.processor.nr_total),
                                int(nl_filter.col.processor.nr_total - self.node[node_idx]))
            self.node[node_idx] += nr_to_process

            if option[0] == 'i':
                # sql_sample = f"""
                # INSERT INTO scores{fid} SELECT img, score, FALSE FROM
                # (SELECT img, ROW_NUMBER() OVER(ORDER BY NULL) AS rid FROM
                #     (SELECT img
                #     FROM (SELECT img, aid, score FROM images LEFT JOIN scores{fid} ON img = scores{fid}.sid) AS images
                #     WHERE score IS NULL
                #     USING SAMPLE {nr_to_process} ROWS)) AS imgs,
                # (SELECT score, ROW_NUMBER() OVER(ORDER BY NULL) AS rid FROM scores{fid} USING SAMPLE {nr_to_process} ROWS) AS samples
                # WHERE imgs.rid = samples.rid"""
                # sql_sids = f"""
                # SELECT pid
                # FROM (SELECT images.pid, score FROM images LEFT JOIN scores ON images.pid = scores.pid) AS t
                # WHERE score IS NULL
                # USING SAMPLE {nr_to_process} ROWS"""
                sql_sids = f"""SELECT sid FROM (SELECT sid FROM scores{fid} WHERE score IS NULL) AS temp_scores USING SAMPLE {nr_to_process} ROWS"""
            elif option[0] == 'o':
                (_, _, col_idx, min_max, _) = option
                col_name = self.query.cols[col_idx]
                table_name = self.nldb.get_table_by_col_from_query(col_name, self.query)
                asc_desc = 'ASC' if min_max == 'min' else 'DESC'
                # sql_sample = f"""
                # INSERT INTO scores{fid} SELECT img, score, FALSE FROM
                # (SELECT img, ROW_NUMBER() OVER(ORDER BY NULL) AS rid FROM
                #     (SELECT img
                #     FROM (SELECT img, aid, score FROM images LEFT JOIN scores{fid} ON img = scores{fid}.sid) AS images,
                #     furniture
                #     WHERE score IS NULL AND furniture.aid = images.aid
                #     ORDER BY {table_name}.{col_name} {asc_desc}
                #     LIMIT {nr_to_process})) AS imgs,
                # (SELECT score, ROW_NUMBER() OVER(ORDER BY NULL) AS rid FROM scores{fid} USING SAMPLE {nr_to_process} ROWS) AS samples
                # WHERE imgs.rid = samples.rid"""

                # TODO: Sampling smaller subset and claim that it is an unbiased estimator.
                # TODO: Just work with smaller subset for simulation.
                # TODO: Physical layout vs logical layout.
                # TODO: Pro-active querying for future simulations.
                sql_sids = self.nldb.get_sql_sids(fid, col_name, table_name, asc_desc, nr_to_process, nl_filter)
                # print(f'sql_sids: {sql_sids}')

                # sql_sids = f"""
                # SELECT img
                # FROM (SELECT img, aid, score FROM images LEFT JOIN scores{fid} ON img = scores{fid}.sid) AS images,
                # furniture
                # WHERE score IS NULL AND furniture.aid = images.aid
                # ORDER BY {table_name}.{col_name} {asc_desc}
                # LIMIT {nr_to_process}"""

            sql_scores = f"""
            SELECT score FROM (SELECT score FROM scores{fid} WHERE score IS NOT NULL) AS temp_scores USING SAMPLE {nr_to_process} ROWS
            """

            # print('processed rows:', nr_to_process)
            # print('before:', self.nldb.con.execute("select count(*) from scores").fetchall()[0][0])
            start = time.time()
            # start_sids = start
            sids = self.nldb.con.execute(sql_sids).df().iloc[:, 0].to_list()
            # end_sids = time.time()
            # global runtimes
            # runtimes['insert_sids'] += end_sids - start_sids
            # start_scores = end_sids
            scores = self.nldb.con.execute(sql_scores).df().iloc[:, 0].to_list()
            # end_scores = time.time()
            # runtimes['insert_scores'] += end_scores - start_scores
            if len(sids) != len(scores):
                # print(f'Lengths of sids and scores are different: {option} {nr_to_process} {len(sids)} {len(scores)}')
                assert len(sids) > len(scores)
                sids = sids[:len(scores)]
            # assert len(sids) == len(scores), f'{option} {nr_to_process} {len(sids)} {len(scores)}'
            assert not any(math.isnan(score) for score in scores)
            assert not any(sid in nl_filter.idx_to_score for sid in sids), f'{option} {sql_sids}'
            # start_update = time.time()
            self.nldb.con.executemany(f"UPDATE scores{fid} SET score=? WHERE sid=?", [[score, sid] for sid, score in zip(sids, scores)])
            # end_update = time.time()
            # runtimes['insert_update'] += end_update - start_update

            # df_pids = self.nldb.con.execute(sql_pids).df()
            # df_scores = self.nldb.con.execute(sql_scores).df()
            # df_concat = pd.concat([df_pids, df_scores], axis=1, copy=False)
            # print(option, len(df_pids), len(df_scores), len(df_concat))
            # self.nldb.con.execute("""INSERT INTO scores SELECT *, FALSE FROM df_concat""")

            # self.nldb.con.execute(sql_sample)
            end = time.time()
            # print('after:', self.nldb.con.execute("select count(*) from scores").fetchall()[0][0])
            global runtimes
            runtimes['insert'] += end - start
            # print(f'Time - Insert more images: {end - start}')

            # score_sample = random.choices(self.score_list, k=nr_to_process)
            # con.executemany("INSERT INTO scores VALUES (?, ?)", [[-1, score] for score in score_sample])
        elif option[0] == 'u':
            weight = option[1]

            node_idx = fid * self.nr_node_vals_per_filter
            lower = self.node[node_idx]
            upper = self.node[node_idx + 1]
            cur_min = max(lower, nl_filter.lower)
            cur_max = min(upper, nl_filter.upper)
            cur_score = cur_min * (1 - weight) + cur_max * weight

            if random.random() < weight:  # Yes
                self.node[node_idx + 1] = cur_score  # Upper threshold
            else:  # No
                self.node[node_idx] = cur_score  # Lower threshold

        # Reward computation.
        start_query = time.time()
        error = self.nldb.query_to_compute_error(self.query, self.nl_filters)
        end_query = time.time()
        # global runtimes
        runtimes['query'] += end_query - start_query
        improvement = self.prev_error - error
        reward = improvement / cost
        self.prev_error = error

        # Check for terminate condition.
        # TODO: or self.node[-1] == self.nl_filter.col.processor.nr_total
        done = bool(error <= self.error_constraint or self.cur_step == self.max_nr_actions)
        # if self.first_action is None:
        #     self.first_action = action
        # if done:
        #     global max_reward_action
        #     if max_reward_action[0] < reward:
        #         max_reward_action = (reward, self.first_action)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.node, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f'obs: {self.node.tolist()}')
        start_query = time.time()
        error = self.nldb.query_to_compute_error(self.query, self.nl_filters, print_error=True)
        end_query = time.time()
        global runtimes
        runtimes['query'] += end_query - start_query
        return error

    def close(self):
        pass

    def action_to_option(self, action):
        fid = math.floor(action / self.nr_actions_per_filter)
        faction = action % self.nr_actions_per_filter
        if faction < self.nr_user_feedback_opts:
            option = ('u', (1 + faction) / (1 + self.nr_user_feedback_opts), fid)
        elif faction == self.nr_user_feedback_opts:
            option = ('i', self.process_percent_multiple, fid)
        else:
            col_ref = faction - (1 + self.nr_user_feedback_opts)
            col_idx = math.floor(col_ref / 2)
            min_max = 'min' if col_ref % 2 == 0 else 'max'
            option = ('o', self.process_percent_multiple, col_idx, min_max, fid)
        return option

    # def recreate_scores_table(self):
    #     start = time.time()
    #     for fid, nl_filter in enumerate(self.nl_filters):
    #         # self.nldb.con.execute(f"DELETE FROM scores{fid} WHERE NOT processed")
    #         self.nldb.con.execute(f"UPDATE scores{fid} SET score=NULL WHERE NOT processed")
    #     end = time.time()
    #     global runtimes
    #     runtimes['recreate'] += end - start
    #     # print(f'Time - Recreate table: {end - start}')

    # def query_to_compute_error(self, print_error=False, print_result=False, return_results=False):
    #     start = time.time()
    #     # Compute lower and upper bounds.
    #     # Lower and upper thresholds of NL predicates are given as a list of tuples, e.g., [(lt, ut), ...]
    #     thresholds = [(self.node[fid * self.nr_node_vals_per_filter], self.node[1 + fid * self.nr_node_vals_per_filter])
    #                  for fid in range(len(self.nl_filters))]
    #     sql_l, sql_u, nr_avgs = self.query.to_lower_upper_sqls(self.nl_filters, thresholds)
    #     # if is_print:
    #     #     print(f'sql_l: {sql_l}')
    #     #     count_sql_l = f'SELECT count(*) FROM scores0 WHERE scores0.score >= {thresholds[0][1]}'
    #     #     print(f'count_sql_l: {self.nldb.con.execute(count_sql_l).fetchall()[0][0]} / {self.nldb.con.execute("SELECT count(*) FROM scores0").fetchall()[0][0]} {count_sql_l}')
    #     con_l = self.nldb.con.execute(sql_l)
    #     result_l = con_l.fetchall()
    #     result_u = self.nldb.con.execute(sql_u).fetchall()
    #     lus = []
    #     if self.query.limit < 0:
    #         nr_selects = len(con_l.description)
    #         nr_non_avgs = nr_selects - 2 * nr_avgs
    #         # Aggregates except avgs.
    #         for idx in range(nr_non_avgs):
    #             select_str, *_ = con_l.description[idx]
    #             if schema.is_aggregate(select_str) and not schema.is_avg_aggregate(select_str):
    #                 l = result_l[0][idx]
    #                 u = result_u[0][idx]
    #                 if l is None or u is None:
    #                     lus.append((float('-inf'), float('inf')))
    #                 else:
    #                     if l > u:
    #                         temp_val = l
    #                         l = u
    #                         u = temp_val
    #                     lus.append((l, u))
    #         for idx in range(nr_non_avgs, nr_selects, 2):
    #             l_s = result_l[0][idx]
    #             u_s = result_u[0][idx]
    #             l_c = result_l[0][idx + 1]
    #             u_c = result_u[0][idx + 1]
    #             if l_s is None or u_s is None or l_c is None or u_c is None or l_c == 0 or u_c == 0:
    #                 lus.append((float('-inf'), float('inf')))
    #             else:
    #                 if l_s > u_s:
    #                     temp_val = l_s
    #                     l_s = u_s
    #                     u_s = temp_val
    #                     temp_val = l_c
    #                     l_c = u_c
    #                     u_c = temp_val
    #                 assert l_c <= u_c
    #                 lus.append((l_s / u_c, u_s / l_c))
    #     else:
    #         l = min(self.query.limit, len(result_l))
    #         u = min(self.query.limit, len(result_u))
    #         lus.append((l, u))
    #     end = time.time()

    #     global runtimes
    #     runtimes['query'] += end - start

    #     if print_result:
    #         print(f'LOWER BOUNDS:\n{result_l}')
    #         print(f'UPPER BOUNDS:\n{result_u}')

    #     errors = [1 if math.isnan(error) else error for error in ((u - l) / (u + l) if l != u else 0 for l, u in lus)]
    #     error_avg = sum(errors) / len(lus)
    #     if print_error:
    #         print(f'Error: {error_avg} {errors}, Bounds: {lus}')

    #     if return_results:
    #         return error_avg, lus

    #     return error_avg

        # # lower_sql = "'-Infinity'" if self.node[0] == float('-inf') else str(self.node[0])
        # # upper_sql = "'Infinity'" if self.node[1] == float('inf') else str(self.node[1])
        # lower_sql = self.node[0]
        # upper_sql = self.node[1]
        # if self.sq.agg_func == 'count':
        #     start = time.time()
        #     nr_unsure = self.nldb.con.execute(f"SELECT count(*) FROM scores WHERE score > {lower_sql} AND score < {upper_sql}").fetchall()[0][0]
        #     nr_true = self.nldb.con.execute(f"SELECT count(*) FROM scores WHERE score >= {upper_sql}").fetchall()[0][0]
        #     end = time.time()
        #     nr_yet = self.nl_filter.processor.nr_total - self.node[-1]
        #     l = min(self.sq.nr_limit, nr_true)
        #     u = min(self.sq.nr_limit, nr_true + nr_unsure + nr_yet)
        # elif self.sq.agg_func == 'min' or self.sq.agg_func == 'max':
        #     start = time.time()
        #     l = self.nldb.con.execute(f"""
        #     SELECT {self.sq.agg_func}({self.sq.agg_col})
        #     FROM (SELECT images.pid, images.aid, score FROM images LEFT JOIN scores ON images.pid = scores.pid) AS t,
        #         furniture
        #     WHERE furniture.aid = t.aid
        #     AND (score IS NULL OR score > {lower_sql})
        #     """).fetchall()[0][0]
        #     u = self.nldb.con.execute(f"""
        #     SELECT {self.sq.agg_func}({self.sq.agg_col})
        #     FROM (SELECT images.pid, images.aid FROM images LEFT JOIN scores ON images.pid = scores.pid
        #         WHERE score >= {upper_sql}) AS t,
        #         furniture
        #     WHERE furniture.aid = t.aid
        #     """).fetchall()[0][0]
        #     if l is None:
        #         l = self.nldb.con.execute(f"SELECT {self.sq.agg_func}({self.sq.agg_col}) FROM furniture").fetchall()[0][0]
        #     if u is None:
        #         opp_func = 'max' if self.sq.agg_func == 'min' else 'min'
        #         u = self.nldb.con.execute(f"SELECT {opp_func}({self.sq.agg_col}) FROM furniture").fetchall()[0][0]
        #     end = time.time()
        #     if self.sq.agg_func == 'max':
        #         temp_val = l
        #         l = u
        #         u = temp_val
        #
        # global runtimes
        # runtimes['query'] += end - start
        # # print(f'Time - Querying True/Unsure: {end - start}')
        # error = (u - l) / (u + l) if l != u else 0
        # if is_print:
        #     if self.sq.agg_func == 'count':
        #         print(f'Error: {error}, True Unsure Yet: {nr_true} {nr_unsure} {nr_yet}, Bounds: {l} {u}')
        #     else:
        #         print(f'Error: {error}, Bounds: {l} {u}')
        # return error
