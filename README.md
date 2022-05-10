<p align="center">
    <br>
    <img src="https://github.com/d0r1h/ILC/blob/main/assets/ILC_logo.png" width="300"/>
    <br>
<p>
    
    
<p align="center">
    <a href="http://pawantrivedi.me/ILC">
        <img alt="Website" src="https://img.shields.io/website? down_color=red&down_message=offline&up_color=yello&up_message=online&url=http%3A%2F%2Fpawantrivedi.me%2FILC%2F">
    </a>
    <a href="https://hits.seeyoufarm.com">
        <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fd0r1h%2FILC&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false">
    </a>    
</p>    

    

    

## DataSet

```python
from datasets import load_dataset

dataset = load_dataset("d0r1h/ILC")

train_set = pd.DataFrame(dataset['train'])
test_set = pd.DataFrame(dataset['test'])
```


## Code

```python
git clone https://github.com/d0r1h/ILC.git
cd ILC
pip install -r requirement.txt
```


Summarzing using Extractive approach 

```python

!python Code/Models/extractive.py \
        --output_dir dir_name \
        --text_column text \
        --summary_column summary \
        --data_file data.csv \
        --sentence_count 3 
```

Training LED using Abstractive approach 

```python
!python Code/Models/led_summarization.py \
        --model_name  allenai/led-base-16384 \
        --text_column  Case \
        --summary_column Summary    \
        --max_input_length  8192 \
        --max_output_length  600 \
        --batch_size 2 \
        --num_beams 2 \
        --output_dir output_dir_name
```

Inference on test-set using [led-base-ilc](https://huggingface.co/d0r1h/led-base-ilc) model

| Notebook | Colab |
| ------ | ------ |
| led-base-ilc | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/ILC/blob/main/Code/Models/notebooks/LEDbase_inference_ilc_testset.ipynb) |


## Results: 

Following results are obtained on test-set with transformer based models and extractive methods

| Algorithm / model | Rouge-1 | Rouge-2 | Rouge-L|
| ---- | ---- | ---- | ----|
**Extractive**
| SumBasics	 |	15.69 | 6.02   | 14.48 |
| LSA 	     |	21.20 | 7.37  | 19.76 |
| KLSum      |	21.40 | 10.19  | 19.66 |
| LexRank 	 |  33.09 | 16.81  | 22.99 |  
| TextRank 	 |  34.54 | 18.10  | 31.11 |
**Abstractive**
| LedBase    | 4.31  | 1.08 | 4.11 |
| Led-ilc     | **42.24** | **23.18**  | **39.30** |

