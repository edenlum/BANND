# BANND

A project for the course "Trustworthy Machine Learning". We investigate defenses for
backdoor attacks on neural networks during training.

# Dependencies

The project requires Python 3.10, and dependencies are installed using `pip`.

To install the dependencies, run `python -m pip install -r requirements`.

We recommend using a Virtual Environment for the project. To do that, install
`virtualenv` with `python -m pip install virtualenv` and create a new virtualenv in the
project with `python -m virtualenv ./venv`. Then, activate the virtualenv with `source
./venv/bin/activate` and run all the `python` and `pip` commands within it.

# Running

Run the project with `python bannd.py`. Pass the `-h`/`--help` flag to see all the
parameters. Note that for brevity's sake, some parameters like the number of epochs and
batch size are omitted and can't be passed.

We support 3 modes of running:

- Baseline: pass `--runtype baseline` to see the network's baseline accuracy without any
  attack or defense.
- Attack: pass `--runtype attack` and `--poison_rate Y` (any other number between
  `0.01` and `0.99`, we tested with `0.1`, `0.3`, and `0.5`) to run the badnets attack on the network, by poisoning some
  percentage of the samples with the backdoor. Expect the attack success rate to
  increase with time.
- Defense: pass `--runtype defense`, `--poison_rate X`, and `--quantile-threshold Y`
  (any number between `0.00` and `0.99`, we tested with `0.0`, `0.5`, `0.7`, and `0.85`)
  to run the attack and defend against it.

## Examples

To run the baseline:

```sh
python bannd.py --runtype baseline
```

To run the attack:

```sh
python bannd.py --runtype attack --poison_rate 0.1
```

To run the defense:

```sh
python bannd.py --runtype defense --poison_rate 0.1 --quantile-threshold 0.7
```
