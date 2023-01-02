Flexible and intuitive build of [MuZero](https://arxiv.org/pdf/1911.08265.pdf).

Currently implemented and trained: TicTacToe, CartPole, CarRacing.

Use [main.py](https://github.com/foersterrobert/AlphaZero/blob/master/main.py) for training your own models.

```python
pip install -r requirements.txt
python main.py
```

(Currently the main.py script is only fully set for TicTacToe. For the other environments please visit the [Sripts](https://github.com/foersterrobert/MuZero/tree/master/Scripts) folder. This will be cleaned up in the future)

Alternatively you can also use the pretrained models in the [Environments](https://github.com/foersterrobert/MuZero/tree/master/Environments) folder and test them in [test.ipynb](https://github.com/foersterrobert/MuZero/blob/master/test.ipynb)
(set num_iterations to 0 to visualize the board).

![carracing](https://raw.githubusercontent.com/foersterrobert/MuZero/master/assets/carracing.gif)
![cartpole](https://raw.githubusercontent.com/foersterrobert/MuZero/master/assets/cartpole.gif)
