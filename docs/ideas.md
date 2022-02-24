# Bug Finding - essentially MCSs

1. The user provides a set of atoms that can be a part of a valid model and should't OR that cannot be part of a valid model but should.
2. The tool uses the linear MCS algorithm from SAT to find a set of restrictions that need to be removed/modified in order to obtain the expected behaviour.

### Next steps
- It's possible that there are several MCSs - returning several to the user might be useful
- After finding the buggy restrictions we can attempt to perform automatic bug fixing

**NOTE: this is pretty basic but I did not find anyone doing it already**

1h07: (or (= (+ queen_A_0 queen_A_1) (+ queen_B_1 queen_B_0)) (or (= queen_A_1 queen_B_1) (= queen_A_0 queen_B_0)))