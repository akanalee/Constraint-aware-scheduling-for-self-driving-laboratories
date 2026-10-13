import itertools

import gurobipy as gp
from gurobipy import GRB

SCALE = 10
BIG = 100000

_ENV = None


def _gurobi_env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam('OutputFlag', 0)
        _ENV.start()
    return _ENV


def check_release_feasibility(env, jobs_to_release, time_limit_sec=0.5,
                              current_time=None, return_status=False):
    # Rapid Gurobi feasibility check of a candidate release subset: the subset is
    # skipped only if no feasible schedule exists for its operations together with
    # the unstarted operations already in the system.
    graph = env.graph
    if current_time is None:
        current_time = env.current_time
    tmax_cstrs = env.tmax_constraints
    machines_ids = [m.id for m in env.machines]

    def _ret(feasible, status):
        return (feasible, status) if return_status else feasible

    existing_ops = {op_id: op for op_id, op in graph.operations.items()
                    if not op.is_completed}

    new_ops = {}
    new_precedences = []
    for job in jobs_to_release:
        for i, op_data in enumerate(job.operations):
            op_id = op_data['id']
            new_ops[op_id] = {
                'id': op_id,
                'job_id': job.id,
                'op_index': i,
                'op_type': op_data.get('op_type', 'default'),
                'processing_times': op_data['processing_times'],
                'arrival_time': job.arrival_time,
                'predecessors': [job.operations[i - 1]['id']] if i > 0 else [],
            }
        for i in range(1, len(job.operations)):
            new_precedences.append((job.operations[i - 1]['id'],
                                    job.operations[i]['id']))

    all_op_ids = list(existing_ops.keys()) + list(new_ops.keys())

    def _get_op_field(op_id, field):
        if op_id in existing_ops:
            op = existing_ops[op_id]
            if field == 'job_id':
                return op.job_id
            if field == 'op_index':
                return op.op_index
            if field == 'op_type':
                return op.op_type
            if field == 'processing_times':
                return op.processing_times
            if field == 'arrival_time':
                return getattr(op, 'arrival_time', 0.0)
            if field == 'predecessors':
                return op.predecessors
            if field == 'is_running':
                return op.is_scheduled and op.scheduled_start <= current_time + 1e-4
            if field == 'scheduled_start':
                return op.scheduled_start
            if field == 'scheduled_end':
                return op.scheduled_end
            if field == 'scheduled_machine':
                return op.scheduled_machine
        else:
            d = new_ops[op_id]
            if field == 'is_running':
                return False
            if field == 'scheduled_start':
                return 0.0
            if field == 'scheduled_end':
                return 0.0
            if field == 'scheduled_machine':
                return None
            return d.get(field)

    mdl = gp.Model('release_feasibility', env=_gurobi_env())
    mdl.setParam('TimeLimit', time_limit_sec)
    mdl.setParam('Threads', 1)

    op_start, op_end = {}, {}
    machine_items = {mid: [] for mid in machines_ids}

    for op_id in all_op_ids:
        proc_times = _get_op_field(op_id, 'processing_times')
        arrival = _get_op_field(op_id, 'arrival_time')
        is_running = _get_op_field(op_id, 'is_running')

        if is_running:
            s_val = int(_get_op_field(op_id, 'scheduled_start') * SCALE)
            e_val = int(_get_op_field(op_id, 'scheduled_end') * SCALE)
            op_start[op_id] = s_val
            op_end[op_id] = e_val
            m_fixed = _get_op_field(op_id, 'scheduled_machine')
            if m_fixed and m_fixed in machine_items:
                machine_items[m_fixed].append((op_id, None, e_val - s_val))
            continue

        lb_scaled = int(max(current_time, arrival) * SCALE)
        s_var = mdl.addVar(lb=lb_scaled, ub=BIG, name=f's_{op_id}')
        e_var = mdl.addVar(lb=lb_scaled, ub=BIG, name=f'e_{op_id}')
        op_start[op_id] = s_var
        op_end[op_id] = e_var

        compat_machines = [mid for mid in machines_ids if mid in proc_times]
        if not compat_machines:
            mdl.dispose()
            return _ret(False, 'infeasible')

        pres = {}
        dur_expr = gp.LinExpr()
        for mid in compat_machines:
            dur_scaled = max(1, int(proc_times[mid] * SCALE))
            p_var = mdl.addVar(vtype=GRB.BINARY, name=f'p_{op_id}_{mid}')
            pres[mid] = p_var
            dur_expr.add(p_var, dur_scaled)
            machine_items[mid].append((op_id, p_var, dur_scaled))
        mdl.addConstr(gp.quicksum(pres.values()) == 1)
        mdl.addConstr(e_var == s_var + dur_expr)

    for mid, items in machine_items.items():
        for (a, pa, _), (b, pb, _) in itertools.combinations(items, 2):
            if pa is None and pb is None:
                continue
            y = mdl.addVar(vtype=GRB.BINARY)
            both_off = 2 - (pa if pa is not None else 1) - (pb if pb is not None else 1)
            mdl.addConstr(op_start[b] >= op_end[a] - BIG * (1 - y) - BIG * both_off)
            mdl.addConstr(op_start[a] >= op_end[b] - BIG * y - BIG * both_off)

    def _add_prec(pred_id, succ_id):
        if pred_id not in op_end or succ_id not in op_start:
            return
        mdl.addConstr(op_start[succ_id] >= op_end[pred_id])

    for op_id, op in existing_ops.items():
        for pred_id in op.predecessors:
            _add_prec(pred_id, op_id)
    for pred_id, succ_id in new_precedences:
        _add_prec(pred_id, succ_id)
    for op_id, d in new_ops.items():
        for pred_id in d['predecessors']:
            if pred_id in existing_ops:
                _add_prec(pred_id, op_id)

    job_type_map = {}
    for op_id in all_op_ids:
        jid = _get_op_field(op_id, 'job_id')
        otype = _get_op_field(op_id, 'op_type')
        oidx = _get_op_field(op_id, 'op_index')
        job_type_map.setdefault(jid, {}).setdefault(otype, []).append((oidx, op_id))

    for (from_step, to_step, max_iv) in tmax_cstrs:
        max_iv_scaled = int(max_iv * SCALE)
        for jid, type_dict in job_type_map.items():
            to_list = type_dict.get(to_step, [])
            from_list = type_dict.get(from_step, [])
            if not to_list or not from_list:
                continue
            for to_idx, to_op_id in to_list:
                cands = [(fi, fid) for fi, fid in from_list if fi < to_idx]
                if not cands:
                    continue
                _, from_op_id = max(cands, key=lambda x: x[0])

                if from_op_id not in op_end:
                    from_op_obj = graph.operations.get(from_op_id)
                    if from_op_obj and (from_op_obj.is_completed or from_op_obj.is_scheduled):
                        deadline_scaled = int((from_op_obj.scheduled_end + max_iv) * SCALE)
                        if to_op_id in op_start:
                            mdl.addConstr(op_start[to_op_id] <= deadline_scaled)
                    continue

                if to_op_id in op_start:
                    mdl.addConstr(op_start[to_op_id] - op_end[from_op_id] <= max_iv_scaled)

    mdl.optimize()
    status = mdl.Status
    has_sol = mdl.SolCount > 0
    mdl.dispose()

    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return _ret(False, 'infeasible')
    return _ret(True, 'feasible' if has_sol else 'unknown')
