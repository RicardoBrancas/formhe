PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 184_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 185_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-non-relaxed --skip-mcs-line-pairings
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 186_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-relaxed --skip-mcs-line-pairings
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 187_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-non-relaxed --skip-mcs-negative-relaxed --use-mcs-positive --skip-mcs-line-pairings
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 188_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-non-relaxed --skip-mcs-negative-relaxed
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 189_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-non-relaxed --skip-mcs-negative-relaxed --skip-mcs-line-pairings --use-mfl
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 190_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --skip-mcs-negative-non-relaxed --skip-mcs-negative-relaxed --skip-mcs-line-pairings --use-sbfl
PYTHONHASHSEED=0 python3 helper_scripts/evaluate.py 191_repeat -m=8192 -p=1 --selfeval-fix-test --disable-classical-negation --allow-unsafe-vars --disable-commutative-predicate --disable-distinct-args-predicate --enable-redundant-arithmetic-ops --disable-head-empty-or-non-constant-constraint --disable-no-dont-care-in-head-constraint
