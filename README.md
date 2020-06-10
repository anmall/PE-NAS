# PE-NAS

This is the repor for paper PE-NAS.

To run it, please install 

1) [NASbench-101](https://github.com/google-research/nasbench) api.
2) [LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

'''
python run_re_nips.py --n_workers 16 --num_process 1 --benchmark nasbench101 --n_iters 5000 --predict_mutate true --predict_kill true
'''

