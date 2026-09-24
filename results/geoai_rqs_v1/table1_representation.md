# Table 1: matched representation transfer

Source/zero-shot reuse 3 frozen RF seeds. Few-shot columns are distributed target-only weak-label RFs, with shared draws and fixed test rows. Source and few-shot tree counts differ as in the original benchmark; compare representations within each condition.

| representation | source_f1_mean | source_f1_std | target_zero_shot_f1_mean | target_zero_shot_f1_std | transfer_drop | source_seeds | f1_50_mean | f1_50_std | n_seeds_50 | f1_100_mean | f1_100_std | n_seeds_100 | f1_250_mean | f1_250_std | n_seeds_250 | f1_500_mean | f1_500_std | n_seeds_500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_reconstructed | 0.9595 | 0.0043 | 0.8149 | 0.0030 | 0.1446 | 3 | 0.8106 | 0.0235 | 30 | 0.8259 | 0.0139 | 30 | 0.8521 | 0.0060 | 5 | 0.8594 | 0.0063 | 30 |
| alphaearth | 0.9648 | 0.0032 | 0.8360 | 0.0024 | 0.1288 | 3 | 0.8706 | 0.0121 | 30 | 0.8774 | 0.0095 | 30 | 0.8930 | 0.0027 | 5 | 0.8981 | 0.0041 | 30 |
| presto_primary | 0.9487 | 0.0056 | 0.8262 | 0.0021 | 0.1225 | 3 | 0.8126 | 0.0438 | 30 | 0.8436 | 0.0257 | 30 | 0.8676 | 0.0128 | 5 | 0.8840 | 0.0068 | 30 |
| galileo | 0.9416 | 0.0015 | 0.8221 | 0.0012 | 0.1195 | 3 | 0.8262 | 0.0292 | 30 | 0.8443 | 0.0127 | 30 | 0.8515 | 0.0202 | 5 | 0.8617 | 0.0056 | 30 |
