# Binary File Visualization with Wavelets

<div style="display:flex;align-items:center;flex-direction:column;">
    <img src="./sample_gdb.png" alt="A visualization sample" style="width:100%;max-width: 400px">

*Above: A sample output visualizing the GDB binary*

</div>

This is a simple file visualization script that uses a 2D wavelet transform to help visualize files in a binary format. The visualization strategy can be adjusted to use several discrete wavelets and 2D mapping methods.

## Setup

Please use Python 3.7 for backward compatibility with some of the requirements.

If you are running into version errors, please use Python 3.7 and the **exact** minimum version specified in the `requirements.txt` file.

```bash
# activate a virtual env before this if desired
pip install -r requirements.txt
```

## Running

```bash
python main.py -o output.png file_to_visualize
```

The arguments available can be viewed with `-h`:

```bash
python main.py -h
```

## Citing
Please use the following citation when using our work.

```
@article{Content-Agnostic-Malware-Identification,
  author = {Nathaniel J. Fernandes and Griffith Thomas and Ben Allin},
  title = {Content-Agnostic Malware Identification and Binary Data Visualization Using Wavelet Transforms},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository. https://github.com/Nathaniel-Fernandes/math-414-final-project},
  howpublished = {https://github.com/Nathaniel-Fernandes/math-414-final-project},
  commit = {X}
}
```


