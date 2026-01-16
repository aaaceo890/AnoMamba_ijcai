import math
from itertools import groupby
from operator import itemgetter

def test_events(events):
    if type(events) is not list:
        raise TypeError('Input `events` should be a list of couples')
    if not all([type(x) is tuple for x in events]):
        raise TypeError('Input `events` should be a list of tuples')
    if not all([len(x) == 2 for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop)')
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop) with start <= stop')
    if not all([events[i][1] < events[i + 1][0] for i in range(len(events) - 1)]):
        raise ValueError('Couples of input `events` should be disjoint and ordered')

def _sum_wo_nan(vec):
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return sum(vec_wo_nan)


def _len_wo_nan(vec):
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return len(vec_wo_nan)

def infer_Trange(events_pred, events_gt):
    if len(events_gt) == 0:
        raise ValueError('The gt events should contain at least one event')
    if len(events_pred) == 0:
        return infer_Trange(events_gt, events_gt)

    min_pred = min([x[0] for x in events_pred])
    min_gt = min([x[0] for x in events_gt])
    max_pred = max([x[1] for x in events_pred])
    max_gt = max([x[1] for x in events_gt])
    Trange = (min(min_pred, min_gt), max(max_pred, max_gt))
    return Trange


def t_start(j, Js=None, Trange=(1, 10)):
    if Js is None:
        Js = [(1, 2), (3, 4), (5, 6)]
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - t_stop(n - 1, Js, Trange)
    else:
        return Js[j][0]

def t_stop(j, Js=None, Trange=(1, 10)):
    if Js is None:
        Js = [(1, 2), (3, 4), (5, 6)]
    if j == -1:
        a = min(Trange)
        return 2 * a - t_start(0, Js, Trange)
    else:
        return Js[j][1]

def E_gt_func(j, Js, Trange):
    range_left = (t_stop(j - 1, Js, Trange) + t_start(j, Js, Trange)) / 2
    range_right = (t_stop(j, Js, Trange) + t_start(j + 1, Js, Trange)) / 2
    return range_left, range_right

def get_all_E_gt_func(Js, Trange):
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return E_gt

def interval_intersection(I=(1, 3), J=(2, 4)):
    if I is None:
        return None
    if J is None:
        return None

    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return None
    else:
        return I_inter_J

def affiliation_partition(Is=None, E_gt=None):
    if E_gt is None:
        E_gt = [(1, 2.5), (2.5, 4.5), (4.5, 10)]
    if Is is None:
        Is = [(1, 1.5), (2, 5), (5, 6), (8, 9)]
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]
        discarded_idx_after = [I[0] > E_gt_j[1] for I in Is]
        kept_index = [not (a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
    return out

def get_pivot_j(I, J):
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')

    j_pivot = None
    if max(I) <= min(J):
        j_pivot = min(J)
    elif min(I) >= max(J):
        j_pivot = max(J)
    else:
        raise ValueError('I should be outside J')
    return j_pivot

def integral_mini_interval(I, J):
    if I is None:
        return 0

    j_pivot = get_pivot_j(I, J)
    a = min(I)
    b = max(I)
    return (b - a) * abs((j_pivot - (a + b) / 2))

def cut_into_three_func(I, J):
    if I is None:
        return None, None, None

    I_inter_J = interval_intersection(I, J)
    if I == I_inter_J:
        I_before = None
        I_after = None
    elif I[1] <= J[0]:
        I_before = I
        I_after = None
    elif I[0] >= J[1]:
        I_before = None
        I_after = I
    elif (I[0] <= J[0]) and (I[1] >= J[1]):
        I_before = (I[0], I_inter_J[0])
        I_after = (I_inter_J[1], I[1])
    elif I[0] <= J[0]:
        I_before = (I[0], I_inter_J[0])
        I_after = None
    elif I[1] >= J[1]:
        I_before = None
        I_after = (I_inter_J[1], I[1])
    else:
        raise ValueError('unexpected unconsidered case')
    return I_before, I_inter_J, I_after

def integral_interval_distance(I, J):
    def f(I_cut):
        return integral_mini_interval(I_cut, J)

    def f0(I_middle):
        return (0)

    cut_into_three = cut_into_three_func(I, J)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right

def interval_length(J=(1, 2)):
    if J is None:
        return 0
    return J[1] - J[0]

def sum_interval_lengths(Is=None):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    return sum([interval_length(I) for I in Is])


def affiliation_precision_distance(Is=None, J=(2, 5.5)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    if all([I is None for I in Is]):
        return math.nan
    return sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(Is)

def interval_subset(I=(1, 3), J=(0, 6)):
    if (I[0] >= J[0]) and (I[1] <= J[1]):
        return True
    else:
        return False

def integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if not interval_subset(J, E):
        raise ValueError('J should be included in E')
    if not interval_subset(I, E):
        raise ValueError('I should be included in E')

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)

    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    C = (1 / 2) * A + m * B
    return (C)

def integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1 / 2) * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)

    DeltaI = i_max - i_min
    DeltaE = e_max - e_min

    output = DeltaI - (1 / DeltaE) * (integral_min_piece + integral_linear_piece + integral_remaining_piece)
    return output

def integral_interval_probaCDF_precision(I, J, E):
    def f(I_cut):
        if I_cut is None:
            return 0
        else:
            return integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E)

    def f0(I_middle):
        if I_middle is None:
            return 0
        else:
            return max(I_middle) - min(I_middle)

    cut_into_three = cut_into_three_func(I, J)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right

def affiliation_precision_proba(Is=None, J=(2, 5.5), E=(0, 8)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    if all([I is None for I in Is]):
        return math.nan
    return sum([integral_interval_probaCDF_precision(I, J, E) for I in Is]) / sum_interval_lengths(Is)


def affiliation_recall_distance(Is=None, J=(2, 5.5)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return math.inf
    E_gt_recall = get_all_E_gt_func(Is, (-math.inf, math.inf))
    Js = affiliation_partition([J], E_gt_recall)
    return sum([integral_interval_distance(J[0], I) for I, J in zip(Is, Js)]) / interval_length(J)

def cut_J_based_on_mean_func(J, e_mean):
    if J is None:
        J_before = None
        J_after = None
    elif e_mean >= max(J):
        J_before = J
        J_after = None
    elif e_mean <= min(J):
        J_before = None
        J_after = J
    else:
        J_before = (min(J), e_mean)
        J_after = (e_mean, max(J))

    return J_before, J_after

def integral_mini_interval_Precall_CDFmethod(I, J, E):
    i_pivot = get_pivot_j(J, I)
    e_min = min(E)
    e_max = max(E)
    e_mean = (e_min + e_max) / 2

    if i_pivot <= min(E):
        return 0
    elif i_pivot >= max(E):
        return 0

    cut_J_based_on_e_mean = cut_J_based_on_mean_func(J, e_mean)
    J_before = cut_J_based_on_e_mean[0]
    J_after = cut_J_based_on_e_mean[1]

    iemin_mean = (e_min + i_pivot) / 2
    cut_Jbefore_based_on_iemin_mean = cut_J_based_on_mean_func(J_before, iemin_mean)
    J_before_closeE = cut_Jbefore_based_on_iemin_mean[
        0]
    J_before_closeI = cut_Jbefore_based_on_iemin_mean[
        1]

    iemax_mean = (e_max + i_pivot) / 2
    cut_Jafter_based_on_iemax_mean = cut_J_based_on_mean_func(J_after, iemax_mean)
    J_after_closeI = cut_Jafter_based_on_iemax_mean[0]
    J_after_closeE = cut_Jafter_based_on_iemax_mean[1]

    if J_before_closeE is not None:
        j_before_before_min = min(J_before_closeE)
        j_before_before_max = max(J_before_closeE)
    else:
        j_before_before_min = math.nan
        j_before_before_max = math.nan

    if J_before_closeI is not None:
        j_before_after_min = min(J_before_closeI)
        j_before_after_max = max(J_before_closeI)
    else:
        j_before_after_min = math.nan
        j_before_after_max = math.nan

    if J_after_closeI is not None:
        j_after_before_min = min(J_after_closeI)
        j_after_before_max = max(J_after_closeI)
    else:
        j_after_before_min = math.nan
        j_after_before_max = math.nan

    if J_after_closeE is not None:
        j_after_after_min = min(J_after_closeE)
        j_after_after_max = max(J_after_closeE)
    else:
        j_after_after_min = math.nan
        j_after_after_max = math.nan

    if i_pivot >= max(J):
        part1_before_closeE = (i_pivot - e_min) * (
                j_before_before_max - j_before_before_min)
        part2_before_closeI = 2 * i_pivot * (j_before_after_max - j_before_after_min) - (
                j_before_after_max ** 2 - j_before_after_min ** 2)
        part3_after_closeI = 2 * i_pivot * (j_after_before_max - j_after_before_min) - (
                j_after_before_max ** 2 - j_after_before_min ** 2)
        part4_after_closeE = (e_max + i_pivot) * (j_after_after_max - j_after_after_min) - (
                j_after_after_max ** 2 - j_after_after_min ** 2)
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    elif i_pivot <= min(J):
        part1_before_closeE = (j_before_before_max ** 2 - j_before_before_min ** 2) - (e_min + i_pivot) * (
                j_before_before_max - j_before_before_min)
        part2_before_closeI = (j_before_after_max ** 2 - j_before_after_min ** 2) - 2 * i_pivot * (
                j_before_after_max - j_before_after_min)
        part3_after_closeI = (j_after_before_max ** 2 - j_after_before_min ** 2) - 2 * i_pivot * (
                j_after_before_max - j_after_before_min)
        part4_after_closeE = (e_max - i_pivot) * (
                j_after_after_max - j_after_after_min)
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    else:
        raise ValueError('The i_pivot should be outside J')

    out_integral_min_dm_plus_d = _sum_wo_nan(out_parts)

    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    C = DeltaJ - (1 / DeltaE) * out_integral_min_dm_plus_d

    return C

def integral_interval_probaCDF_recall(I, J, E):
    def f(J_cut):
        if J_cut is None:
            return 0
        else:
            return integral_mini_interval_Precall_CDFmethod(I, J_cut, E)

    def f0(J_middle):
        if J_middle is None:
            return 0
        else:
            return max(J_middle) - min(J_middle)

    cut_into_three = cut_into_three_func(J, I)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right

def affiliation_recall_proba(Is=None, J=(2, 5.5), E=(0, 8)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return 0
    E_gt_recall = get_all_E_gt_func(Is, E)
    Js = affiliation_partition([J], E_gt_recall)
    return sum([integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]) / interval_length(J)

def has_point_anomalies(events):
    if len(events) == 0:
        return False
    return min([x[1] - x[0] for x in events]) == 0

def pr_from_events(events_pred, events_gt, Trange):
    test_events(events_pred)
    test_events(events_gt)

    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')

    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    if Trange is None:
        raise ValueError('Trange should be indicated (or inferred with the `infer_Trange` function')

    E_gt = get_all_E_gt_func(events_gt, Trange) # divide seamless region for each event
    aff_partition = affiliation_partition(events_pred, E_gt)

    d_precision = [affiliation_precision_distance(Is, J) for Is, J in zip(aff_partition, events_gt)]

    d_recall = [affiliation_recall_distance(Is, J) for Is, J in zip(aff_partition, events_gt)]

    p_precision = [affiliation_precision_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    p_recall = [affiliation_recall_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]
    p_recall_average = sum(p_recall) / len(p_recall)

    dict_out = dict({'precision': p_precision_average,
                     'recall': p_recall_average,
                     'individual_precision_probabilities': p_precision,
                     'individual_recall_probabilities': p_recall,
                     'individual_precision_distances': d_precision,
                     'individual_recall_distances': d_recall})
    return dict_out

def convert_vector_to_events(vector=None):
    if vector is None:
        vector = [0, 1, 1, 0, 0, 1, 0]
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))

    events = [(x, y + 1) for (x, y) in events]

    return events


def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P, R, F