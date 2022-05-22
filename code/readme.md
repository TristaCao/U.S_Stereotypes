Here are the codes to measure stereotypes in pre-trained masked language models. We provide two measurements, including [log probability score (ILPS)](https://arxiv.org/abs/1906.07337) and our proposed approach -- sensitivity test (SeT). In our paper, we also include results using measurement [CEAT](https://arxiv.org/abs/2006.03955), which we simply adopted [their code](https://github.com/weiguowilliam/CEAT).

Run `test_templates.py` to measure group and trait associations (stereotype). Measurement resutls will be saved in the `results/` directory.
Note that you can specify which measurements to use (`line 124`). You can also modify the pre-trained masked language model you wish to test on (`line 96-102`) or play around with different traits and groups.

`quick_set.py` implements the SeT measuring approach as described in our paper. 

`base_ilps.py` implements the original ILPS measureing approach as described in their paper. 

`mod_ilps.py` implements the modified ILPS measureing approach that handles multiple subwords as described in our paper. 
