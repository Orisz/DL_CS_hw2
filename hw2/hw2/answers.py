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
1. We can clearly see that we succeeded in training with L=2,4.
Both L=2 and l=4 got similar test accuracy results (~65%), but L=4 was bit better.
That may be because the model had more channels, meaning more features to learn.
We can also deduce that the important feature were also learned with L=2, and that with L=4 
the network learned more fine features that helped it to get a bit better results.
2. We can see that the L=8 and L=16 failed. That may be because of the vanishing Gradients effect.
The depth of the networks is considerably bigger than the L=2 and L=4.
Considering that is really the problem, using a residual network might help, as it will help to prevent 
the gradients from disappearing all the together.  Also, it might help if we set the hyper-parameters a bit differently,
which might help us to prevent theme from disappearing.    
 

"""


part3_q2 = r"""
In this case, it is again clear that only with L=2 or L=4 the model is trainable.
Particularly, with L=8 we failed to learn, with all the values of K.
This leads us to believe that our conclusion in the previous part was correct.
With L=2, K=32 and K=256 got better results than K=256 and K=128.
The accuracy is about the same as in the previous experiment.
With K=4 however K=65 and K=256 were just as good as the other two values, and even a bit better.
"""

part3_q3 = r"""
Whenever the depth was to big the models became un-trainable.
That may be because of the growing Kernel number.
At the beginning of each layer we have only a few Kernels, and than the number grows drastically. 
It may be that during the transition between layers, the change from a very big number of descriptors
to a very small number of kernel makes it difficult for the network to identify the most important descriptors.
This would also explain why the model of L=1 was trained do well.
"""

part3_q4 = r"""
Using the skip connection got us much better results than the previous experiments.
1.1:
Our accuracy results were better the experiment 1.1. Moreover, in 1.1 experiment the L=8
model was un-trainable, wheres in this experiment L=8 was trained.
It got poorer results than L=2 and L=4, but was trainable.
Using the residual network might have helped with the accuracy as it minimizes the effect to the
disappearing gradients.

1.3:
Here we can see even more clearly that all the models that were not trainable in 1.3 were indeed 
trainable with the residual network. This leads us to believe that 2 things:
firstly - our prior conclusion may have been correct. Using the residual network allowed each layer
to learn from a "less effected" info and thus could extract features on it's own, even if it
was difficult to extract them from the previous layer's values.
secondly - in the prior experiment we have suffered of the vanishing gradients problem too,
which the residual network helped to solve.  
The accuracy results were even better than with L=1 in 1.3, which is reasonable as L=1 is a really
shallow architecture.

"""

part3_q5 = r"""
We used layers of: (conv->relu->BN) and once every 2 of these we added a max-pull layer.
We used a dropout of 0.5 as the previous experiments led us to believe that we might be overfiting and we
wanted to generalize our network.
We also added BN in order to try and speed up the learning process.

We got really got accuracy results on the training set, which leads us to believe that we didn't solve our
generalization problem.
Moreover, with our architecture the test accuracy was a tad "jumpier" linear than in the other experiments at first,
but got to a steady test-accuracy after a bit more epochs than in experiment 1.
This means that our new network might need a different learning rate than the other networks we incountered
in this exercise.
 
"""
# ==============
