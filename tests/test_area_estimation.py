import numpy as np
from src.area_estimation.run_demo import adjusted_proportion
def test_perfect_map_adjustment():
 m=np.r_[np.zeros(100,dtype=int),np.ones(100,dtype=int)];e,b,w,r=adjusted_proportion(m,m,{0:900,1:100},np.random.default_rng(1),100);assert abs(e-.1)<1e-9;assert r=={0:0.0,1:1.0}
