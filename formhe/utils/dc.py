#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## dc.py
##
##  Created on: Dec 21, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

from functools import reduce


def combine(diags):
    """
        Combine the sets of individual diagnoses.
    """

    def combine_recursive(i, prefix):
        """
            Recursive call to combine diagnoses.
        """

        if i == len(diags):
            cdiags.add(tuple(sorted(set(prefix))))
            return

        for d in diags[i]:
            combine_recursive(i + 1, prefix + d)

    # common diagnoses
    cdiags = set([])

    # it may be a bad idea to use recursion here
    combine_recursive(0, [])

    return sorted(cdiags, key=lambda d: len(d))


def combine_improved(diags):
    """
        First, apply the improvement step and then the combination.
        This procedure makes the check of Proposition 4 in the memo.
    """

    filt = [[] for dd in diags]
    diags = [[set(d) for d in dd] for dd in diags]

    cdiags1 = set([])

    for i, dd1 in enumerate(diags):
        for d1 in dd1:
            for j, dd2 in enumerate(diags):
                if i != j:
                    for d2 in dd2:
                        if d1.issuperset(d2):
                            break
                    else:
                        filt[i].append(list(d1))
                        break
            else:
                cdiags1.add(tuple(sorted(d1)))

    # some of the observation may end up having no diagnoses
    filt = list(filter(lambda dd: dd, filt))
    #
    # in this case, do no further combinations
    cdiags2 = combine(filt) if len(diags) == len(filt) else []

    return sorted(set(list(cdiags1) + cdiags2), key=lambda d: len(d))


def filter_garbage(cdiags):
    """
        Apply subsumption operations to get rid of garbage diagnoses.
    """

    def process_diag(processed_db, cl):
        for c in processed_db:
            if c <= cl:
                break
        else:
            processed_db.append(cl)
        return processed_db

    # applying subsumption to get a reduced set of diagnoses
    rdiags = reduce(process_diag, [set(d) for d in cdiags], [])

    return [sorted(d) for d in rdiags]


if __name__ == '__main__':
    diags = [[[0, 1]], [[0], [1], [2]]]

    print(diags)

    print(combine_improved(diags))

# if __name__ == '__main__':
#
#         print('c # of idiags:', sum([len(d) for d in diags]))
#
#         # removing duplicates
#         diags = list(set([tuple(sorted(dd)) for dd in diags]))
#         diags = [[list(d) for d in dd] for dd in diags]
#
#         # combining individual diagnoses
#         if improved:
#             # improved combination
#             if verbose:
#                 print('c applying improved combination')
#
#             cdiags = combine_improved(diags)
#         else:
#             # standard, expensive combination
#             if verbose:
#                 print('c applying standard combination')
#
#             cdiags = combine(diags)
#
#         if verbose > 1:
#             for d in cdiags:
#                 print('c C: {0} 0'.format(' '.join([str(i) for i in d])))
#
#         print('c # of cdiags:', len(cdiags))
#
#         rdiags = filter_garbage(cdiags)
#
#         if verbose:
#             for d in rdiags:
#                 print('c D: {0} 0'.format(' '.join([str(i) for i in d])))
#
#         print('c # of diags:', len(rdiags))
#         print('c # of garbage:', len(cdiags) - len(rdiags))
