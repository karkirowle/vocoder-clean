## Articulatory vocoder

Code for reproducing results in ["Paper to be released"](Link of the paper)

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.8) and (Open)MPI

Run
```
pip install -r requirements.txt
```

Reproduce the cross-validation metric in the paper
```
python preprocessing3.py
python cv.py
```

