r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    #wstd, lr, reg = 0, 0, 0
    wstd, lr, reg = 0.5, 0.055, 0.1
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    #wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.01, 0.005, 0.00003, 0.001
    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    wstd, lr = 0.01, 0.0005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
The results match to what we expected partiatly.

Lets observe the **'train acc' graph**:

Drop=0 got the second highest train acc since the model is so big it is able to memorize the entire train set when going through 30 epocs. Drop=0.4 got lowest train acc by far. since we drop about 40% of our neurons each forward pass it is unable to memorize the train set which we hope will force him to learn to generalize instead and lead to better preformences on the test set.
The surprising result is the train acc when using drop=0.8. We belived this rate of drop will give a very poor train results w.r.t. to the drop=0.4 / 'no drop' rate. A reasonalbe explenation is that the model is now so small at each forward pass so the relative affect on each neuron is huge since there are so little of them. as a result the model is overfitting very quikly w.r.t to the drop=0.4 model, meaning train acc improve where test does not.

Lets observe the **'test acc' graph**:

Here the results are as expected.
As expected the results on the drop=0.4 model are the best. As we wished droping in this rate prevented the model from memorizing the dataset and lead us to better generalization. still the curve is a bit jagged so we might need to tweak down the lr.
For both the 'no drop' and drop=0.8 models the test acc is very poor. The reason for the 'no drop' model is arguably overfitting. To support this claim we notice the huge gap between the tarin acc & test acc graphes(~81%(train) vs. ~21%(test)).

The explanation for the drop=0.8 is more interesting. Since the model at each forward pass is so small it is unable to generalize. So overall when testing it on new data the model tends to be 'indecisive'. i.e. some parts were overfitted at training to some direction where others to different directions(and the process continues until epoch 30).
To support this claim we notice how the test curve(drop=0.8) is extremely jagged around the \~21% accuracy where the 'no drop' curve (at test acc) is steady around this value since it overfited the data with every neurons available to her. Also we are considering huge gap between the tarin acc & test acc graphes(\~86%(train) vs. ~24%(test)).



"""

part2_q2 = r"""
Yes this phenomenon is possible. Lets assume animal classes for classification problem. Now consider we have a cat image with prob of 0.6 to be horse and 0.4 prob for cat and alot of other dogs images with prob 0.9 for dog and 0.1 for horse. Now lets make an optimzer.step() to update the weights of the net.
As a result our cat image outputs prob of 0.3 for horse and 0.7 for cat(better accuracy). BUT it seems that all the dog images after the optimizer step output 0.8 for dog and 0.2 for horse meaning entropy loss increased (disorder increased) but accuracy for the dog images remained the same since we still pick the label with the hightest distribution(p(dog)=0.8). So, overall in this example **accuracy improved** due to the cat image now calssified correctly BUT **cross entropy loss also increased** due to the probability changes in all those dog images.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
