<p align="center">
    <br>
    <img src="https://github.com/d0r1h/ILC/blob/main/assets/ILC_logo.png" width="300"/>
    <br>
<p>


# ILC

This page contains the code and track my work on Indian Legal Document (ILC) summarization. 




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
