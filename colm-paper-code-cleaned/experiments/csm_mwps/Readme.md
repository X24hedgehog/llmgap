# Cognitive Student Models on Math Word Problems

## Generating Datasets
### Setup
While in repo-folder:
```
conda create --name csm-mwps \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate csm-mwps

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install -e mathgap

cd experiments/csm_mwps
pip install -r requirements.txt
```

Example:
```
python generate.py gen-ci-distractors -f data/datasets/ci/distractors.json -n 1000 -i 5
```

### Datasets from Paper
```
python generate.py gen-ci-distractors -f data/datasets/ci/icl/ls100/distractors.json -n 100 -i 11 --inclrt -vi all -va True --mindepth 4 --depthdecay 1.33 --consprob --ruleset nocompeq -opp
cp data/datasets/ci/icl/ls100/distractors.json data/datasets/ci/lp/ls100/distractors.json

python generate.py gen-ci-distractors -f data/datasets/ci/icl/ls500/distractors.json -n 500 -i 4 --inclrt -vi all -va True --mindepth 4 --depthdecay 1.33 --consprob --ruleset nocompeq -opp

cp data/datasets/ci/icl/ls500/distractors.json data/datasets/ci/sol/ls500/distractors.json
python generate.py gen-ci-distractors -f data/datasets/ci/sol/ls100/distractors_trans_part.json -n 100 -i 3 --inclrt -vi all --depthdecay 1.33 --consprob --ruleset transpart --misconruleset keyword

python generate.py gen-am-distractors -f data/datasets/am/icl/100/distractors.json -n 100 -i 11 --inclrt -vi all --mindepth 1 --depthdecay 2
python generate.py gen-am-distractors -f data/datasets/am/icl/500/distractors.json -n 500 -i 11 --inclrt -vi all --mindepth 1 --depthdecay 2
```

#### Llama2 7B Chat
TODO: change model path
```
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_mwp_nort.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_mwp_nort.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "MWP" -rt False --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_mwp_nort_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False -hle True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_mwp_nort_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "MWP" -rt False -hle True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_mwp.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_mwp.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "MWP" -rt True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_mwp_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -hle True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_mwp_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "MWP" -rt True -hle True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_mwp_rtfe.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True --maxnewtokens 256
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/llama2_7B_I_s0s1s3s5_mwp.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 3 -s 5 -es "MWP"

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_instva_nort.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_instva_nort.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "INSTVA" -rt False --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_instva_nort_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False -hle True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_instva_nort_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "INSTVA" -rt False -hle True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_instva.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_instva.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "INSTVA" -rt True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_instva_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -hle True --maxnewtokens 256
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama2_7B_I_s3_instva_hle.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 -es "INSTVA" -rt True -hle True --maxnewtokens 1024
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0-5s10_instva_rtfe.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -rtfe True --maxnewtokens 256
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/llama2_7B_I_s0s1s5_instva.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 1 -s 5 -es "INSTVA"

python generate.py solutions -f data/datasets/ci/sol/ls500/distractors.json -o data/datasets/ci/sol/ls500/llama2_7B_I.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 3 --examplestrategy FILE --examplesfile data/datasets/ci/sol/ls100/distractors_trans_part.json --maxnewtokens 1024
```

#### Llama3.1 8B Instruct
TODO: change model path
```
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_mwp_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "MWP" -rt True
python generate.py icl -cat am -f data/datasets/am/icl/500/distractors.json -o data/datasets/am/icl/500/llama31_8B_I_s3_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_mwp_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "MWP" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp_rtfe.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/llama31_8B_I_s0s1s3s5_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 3 -s 5 -es "MWP"

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_instva_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_instva_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_instva.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_instva.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_instva_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_8B_I_s3_instva_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 3 -es "INSTVA" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_instva_rtfe.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -rtfe True
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/llama31_8B_I_s0s1s5_instva.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 5 -es "INSTVA"
```

#### Llama3.1 70B Instruct
TODO: change model path
```
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_mwp_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_mwp_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt False
python generate.py icl -cat am -f data/datasets/am/icl/500/distractors.json -o data/datasets/am/icl/500/llama31_70B_I_s3_mwp_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt False -hle True
python generate.py icl -cat am -f data/datasets/am/icl/500/distractors.json -o data/datasets/am/icl/500/llama31_70B_I_s3_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt True
python generate.py icl -cat am -f data/datasets/am/icl/500/distractors.json -o data/datasets/am/icl/500/llama31_70B_I_s3_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_mwp_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_mwp_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt True -hle True
python generate.py icl -cat am -f data/datasets/am/icl/500/distractors.json -o data/datasets/am/icl/500/llama31_70B_I_s3_mwp_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "MWP" -rt True -hle True
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/llama31_70B_I_s0s1s3s5_mwp.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 3 -s 5 -es "MWP"

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_instva_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_instva_nort.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_instva.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_instva.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_instva_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/llama31_70B_I_s3_instva_hle.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 3 -es "INSTVA" -rt True -hle True

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_70B_I_s0-5s10_mwp_rtfe.json -m "/cluster/scratch/yanickz/models/meta-llama/Llama-3.1-70B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True

```

#### Qwen2.5 7B Instruct
```
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_mwp_nort.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_mwp_nort.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_mwp_nort_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "MWP" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_mwp.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_mwp.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_mwp_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_mwp_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "MWP" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_mwp_rtfe.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/qwen25_7B_I_s0s1s3s5_mwp.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 3 -s 5 -es "MWP"

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_instva_nort.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_instva_nort.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "INSTVA" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_instva_nort_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "INSTVA" -rt False -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_instva.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_instva.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "INSTVA" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_instva_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls500/distractors.json -o data/datasets/ci/icl/ls500/qwen25_7B_I_s3_instva_hle.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 -es "INSTVA" -rt True -hle True
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen25_7B_I_s0-5s10_instva_rtfe.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INSTVA" -rt True -rtfe True
python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/qwen25_7B_I_s0s1s5_instva.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 0 -s 1 -s 5 -es "INSTVA"

python generate.py solutions -f data/datasets/ci/sol/ls500/distractors.json -o data/datasets/ci/sol/ls500/qwen25_7B_I.json -m "/cluster/scratch/yanickz/models/Qwen/Qwen2.5-7B-Instruct" -s 3 --examplestrategy FILE --examplesfile data/datasets/ci/sol/ls100/distractors_trans_part.json
```

# TEST
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/qwen2_05B_I_s0-5s10_mwp_rtfe.json -m "Qwen/Qwen2-0.5B" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True

python generate.py logprobs -f data/datasets/ci/lp/ls100/distractors.json -o data/datasets/ci/lp/ls100/qwen2_05B_I_s0-5s10_instva.json -m "Qwen/Qwen2-0.5B" -s 0 -s 1 -s 5 -es "INSTVA"



# OLD

### Perplexity Experiment
```
python generate.py gen-ci-distractors -f data/datasets/ci/perplexity/distractors.json -n 100 -i 1 --inclrt
python generate.py logprobs -f data/datasets/ci/perplexity/distractors.json -o data/datasets/ci/perplexity/distractors_logprobs.json -m "meta-llama/Llama-3.1-8B-Instruct"
```


## Experiments
### Generate Datasets
```
python generate.py gen-ci-distractors -f data/datasets/ci/icl/100/distractors.json -n 100 -i 11 --inclrt -vi all
python generate.py gen-ci-distractors -f data/datasets/ci/icl/s100/distractors.json -n 100 -i 11 --inclrt -vi all --depthdecay 1.33 --nonlinearityuniformprob 5
python generate.py gen-ci-distractors -f data/datasets/ci/icl/ls100/distractors.json -n 100 -i 11 --inclrt -vi all --mindepth 4 --depthdecay 1.33 --consprob --ruleset nocompeq
python generate.py gen-ci-distractors -f data/datasets/ci/icl/500/distractors.json -n 500 -i 11 --inclrt -vi all
```

### Generating Plausible Distractors By Following Student Traces

```
python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_8B_I_s0-5s10_vi-all.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_inst.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_8B_I_s0-5s10_vi-all_nort.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_inst_nort.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt False


python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/qwen2_7B_I_s0-5s10_vi-all.json -m "Qwen/Qwen2-7B-Instruct" -s 5 -es "INST" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/qwen2_7B_I_s0-5s10_vi-all_nort.json -m "Qwen/Qwen2-7B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt False


python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_70B_I_s5_vi-all.json -m "meta-llama/Llama-3.3-70B-Instruct" -s 5 -es "INST" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_70B_I_s5_vi-all_nort.json -m "meta-llama/Llama-3.3-70B-Instruct" -s 5 -es "INST" -rt False


python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_8B_I_s0-5s10.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_mwp.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True
python generate.py icl -cat ci -f data/datasets/ci/icl/100/distractors.json -o data/datasets/ci/icl/100/llama31_8B_I_s0-5s10_nort.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_mwp_nort.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt False

```



### LLama 3.1 8B Instruct vs LLama 2 7B Instruct
```
python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama31_8B_I_s0-5s10_mwp.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True

python generate.py icl -cat ci -f data/datasets/ci/icl/ls100/distractors.json -o data/datasets/ci/icl/ls100/llama2_7B_I_s0s5_mwp.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" -s 0 -s 5 -es "MWP" -rt True --maxnewtokens 512
```

##### LLama 3.1 70B Instruct
```
python generate.py icl -cat ci -f data/datasets/ci/icl/500/distractors.json -o data/datasets/ci/icl/500/llama31_70B_I_s5.json -m "meta-llama/Llama-3.1-70B-Instruct" -s 5 -es "MWP" -rt True
```

```
python generate.py icl -cat ci -f data/datasets/ci/icl/500/distractors.json -o data/datasets/ci/icl/500/llama31_70B_I_s5_nort.json -m "meta-llama/Llama-3.1-70B-Instruct" -s 5 -es "MWP" -rt False
```

Then load the output files in icl.ipynb

#### Perplexity Analysis
```
cp data/datasets/ci/icl/100/distractors.json data/datasets/ci/perplexity/distractors.json
```

```
python generate.py logprobs -f data/datasets/ci/perplexity/distractors.json -o data/datasets/ci/perplexity/llama31_8B_I_s0s1s5_vi-all.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 5 -es "INST"
```

Then load the output files in perplexity.ipynb



### Showing RTs inlc First Error
```
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_inst_rtfe.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "INST" -rt True -rtfe True

python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_mwp_rtfe.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -rtfe True
```

python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/qwen2_05B_I_s0s3_inst_rtfe.json -m "Qwen/Qwen2-0.5B" -s 0 -s 3 -es "INST" -rt True -rtfe True


### Show a lot of distractor values for INST
```
python generate.py gen-ci-distractors -f data/datasets/ci/icl/s100/distractors_manyinst.json -n 100 -i 31 --inclrt -vi all --depthdecay 1.33 --nonlinearityuniformprob 5

python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors_manyinst.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0s5s10s20s30_inst_nort.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 5 -s 10 -s 20 -s 30 -es "INST" -rt False

```

### Show High-Level Errors
```
python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/llama31_8B_I_s0-5s10_mwp_hle.json -m "meta-llama/Llama-3.1-8B-Instruct" -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 -s 10 -es "MWP" -rt True -hle True

python generate.py icl -cat ci -f data/datasets/ci/icl/s100/distractors.json -o data/datasets/ci/icl/s100/qwen2_05B_I_s0s3_mwp_hle.json -m "Qwen/Qwen2-0.5B" -s 0 -s 3 -es "MWP" -rt True -hle True
```

### Misconceptions When Solving
```
python generate.py gen-ci-distractors -f data/datasets/ci/sol/ks100/distractors.json -n 100 -i 6 --inclrt -vi all --depthdecay 1.33 --nonlinearityuniformprob 5 --consprob --misconruleset keywords

python generate.py solutions -f data/datasets/ci/sol/ks100/distractors.json -o data/datasets/ci/sol/ks100/llama31_8B_I.json -m "meta-llama/Llama-3.1-8B-Instruct"

python generate.py solutions -f data/datasets/ci/sol/ks100/distractors.json -o data/datasets/ci/sol/ks100/llama31_70B_I.json -m "meta-llama/Llama-3.1-70B-Instruct"
```


python generate.py solutions -f data/datasets/ci/sol/ks100/distractors.json -o data/datasets/ci/sol/ks100/qwen2_05B_I.json -m "Qwen/Qwen2-0.5B"


```
python generate.py gen-ci-distractors -f data/datasets/ci/sol/s100/distractors.json -n 100 -i 6 --inclrt -vi all --depthdecay 1.33 --nonlinearityuniformprob 5 --consprob

python generate.py solutions -f data/datasets/ci/sol/s100/distractors.json -o data/datasets/ci/sol/s100/llama31_8B_I.json -m "meta-llama/Llama-3.1-8B-Instruct"
```


```
python generate.py gen-ci-distractors -f data/datasets/ci/sol/ls100/distractors_trans_part.json -n 100 -i 3 --inclrt -vi all --depthdecay 1.33 --consprob --ruleset transpart --misconruleset keyword

python generate.py gen-ci-distractors -f data/datasets/ci/sol/ls100/distractors.json -n 100 -i 6 --inclrt -vi all --mindepth 4 --depthdecay 1.33 --consprob --ruleset nocompeq

python generate.py solutions -f data/datasets/ci/sol/ls100/distractors.json -o data/datasets/ci/sol/ls100/llama31_8B_I.json -m "meta-llama/Llama-3.1-8B-Instruct" --examplestrategy FILE --examplesfile data/datasets/ci/sol/ls100/distractors_trans_part.json

python generate.py solutions -f data/datasets/ci/sol/ls100/distractors.json -o data/datasets/ci/sol/ls100/llama2_7B_I.json -m "/cluster/work/sachan/foundation_models/Llama-2-7b-chat-hf" --examplestrategy FILE --examplesfile data/datasets/ci/sol/ls100/distractors_trans_part.json --maxnewtokens 512

```


python generate.py solutions -f data/datasets/ci/sol/ls100/distractors.json -o data/datasets/ci/sol/ls100/qwen2_05B_I.json -m "Qwen/Qwen2-0.5B" --examplestrategy FILE --examplesfile data/datasets/ci/sol/ls100/distractors_trans_part.json