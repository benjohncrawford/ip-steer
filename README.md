---
title: "My Research Paper"
bibliography: references.bib
link-citations: true
---

# IPSteer: A Interior Point Based LLM Steering Method
In this repo we implement a novel method of steering LLMs that frames the problem of activation steering as a constrained optimization problem and then attempt to solve that optimization problem using the interior point method. This results in a method that is guaranteed to provide activations in a known safety region that remain as close to possible as the originals, which preserves both the original fluency and high level semantic meaning of the activation. 

To see the results of this method please look at this notebook:
To read more about the method and background see below. 

## Credit
It is very important to note that this code is a direct fork from the work of the authors of "ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment" please find their paper here:
[Paper](https://openreview.net/forum?id=CFewUmgIIL), [Website](https://odesteer.github.io/)
and the original code here:
[Code](https://github.com/ZhaoHongjue/odesteer)

The goal of this repo is to build on the framework they implemented and create our own method for steering LLMs. 

# Motivation

Users of LLMs expect outputs that are generated quickly but are
simultaneously helpful, truthful, intelligible, and friendly. However,
training these models can be extremely expensive and therefore it is
desirable to be able to improve their outputs without going through an
expensive retraining process [^1].
To that end, steering has been a highly researched method of achieving
this goal [^7]. Steering is the process of in some way
modifying the activations of the model at generation time in order to
improve the output. As the goal is to generate responses as quickly as
possible but also ensure that they embody the qualities listed above we
can frame this as a constrained optimization problem.

# Background

## Activation Steering

The activations of an LLM are the dynamic values contained in each
neuron at inference time; it is common to take them as a vector of the
values of each neuron in a given layer in the network. In this way, we
can think of them as a point in $\mathbb{R}^d$ space. Activation
steering is a training free method of improving the quality of responses
of LLMs. The high level idea is to at inference/generation time modify
the activations of a specific layer in the model such that the output
better meets some criteria. There have been many proposed methods of how
exactly to edit these activations.

## Prior Work

Previous work in this field has primarily focused on single step
activation steering in which a precalculated vector is added to the
activations at a specific layer[^2]. The methods of
calculating these vectors vary, but in general they require a dataset of
contrastive pairs, one being an example of a positive generation
matching the expected output and the other being a negative output that
should be avoided. One popular and illustrative method is Contrastive
Activation Addition (CAA), which works by learning a steering vector by
calculating the mean difference between the positive and negative
activations [^3]. One major
problem with using a constant steering vector is it does not take into
account complex patterns in the activation space. Another potential
problem with one step steering is that if steering toward one positive
region, you may be in fact steering away from a different region that is
associated with a different desired trait.

One inspirational paper that shifts the paradigm of single step steering
is ODESteer by Zhao et. al which proposed thinking of steering methods
as solutions to ordinary differential equations
[^6]. They elaborate that by
constructing a barrier function in specific ways this view can
encapsulate basically all previous steering methods. They go on to
propose a specific barrier function which is based on modeling the
probability ratio between the likelihood a given activation comes from
the region of positive activations over the likelihood it came from the
negative region using logistic regression. Two key issues with
this method are that it is unknown how many steps it would take to reach
the safety region, and when the steering terminates we have no guarantee
that the resulting activation is in it.

## Interior Point Method

The interior point method is a second order optimization method for
constrained optimization problems. It works by first starting from a
known feasible point and repeatedly applying Newton's method to a
modified version of the original objective function that includes a
barrier term representing the constraint set [^5]. For an
optimization problem such as:

$$\begin{aligned}
\min_{a \in \mathbb{R}^d} \quad & f(a) \\
\text{subject to} \quad& h(a) \le 0
\end{aligned}$$
We define the log barrier function to be: $$\begin{align*}
    F(a) = \log(h(a))
\end{align*}$$ The augmented problem is then:

$$\begin{aligned}
\min_{a \in \mathbb{R}^d} \quad & f_\eta(a) \\
\end{aligned}$$
where $f_\eta = \eta f(a) - F(a)$ The interior point method works by
solving this augmented problem for increasing values of $\eta$.
Specifically, the algorithm is as follows: We begin from a feasible
point $a_{feasible}$ and $\eta_0 > 0$. For $t = 0, ..., T$ we:

1.  Take one newton step: $$\begin{align*}
        a_{t+1} = a_t + n_{\eta_t}(a_t)
    \end{align*}$$ where $n_{\eta_t}(a_t)$ is the Newton step:
    $$\begin{align*}
        n_{\eta}(a) = -(\nabla^2f_\eta(a))^{-1}\nabla f_\eta(a)
    \end{align*}$$

2.  Update $\eta$: $$\begin{align*}
        \eta_{t+1} = \eta_t\cdot \delta
    \end{align*}$$ where $\delta > 1$ is a predetermined constant step
    size.

We can frame the problem of LLM steering as a constrained optimization
problem by defining the barrier function described in the ODESteer
method described above as a constraint set. We can then define an
objective function, and optimize to find a point that is guaranteed to
be within the barrier function that minimizes that objective function.
By framing the problem in this we can have theoretical guarantees on
both the convergence rate and the satisfaction of the constraints. This
solves two of the key issues listed above with the prior work. Another
benefit is we can expand this definition to handle multiple constraints
quite easily, and with this formulation we can handle multiple safety
requirements simultaneously, which solves yet another issue with the
prior work.

## Objective Function

One key concern of steering is ensuring that the output is still
intelligible and on topic compared to what the output would have been
without steering. A reasonable assumption is that activations that are
nearby to each other in activation space have similar meanings and
further that if we don't edit the response by much that it will remain
on-topic. Therefore, we propose the following
objective function:

$$\begin{aligned}
\min_{a \in \mathbb{R}^d} \quad & \frac{1}{2}||a - a_0||_2^2 \\
\end{aligned}$$

Where $a_0$ is defined to be the activation before steering. Using this
objective function will ensure we find the point that satisfies the
constraints while remaining as close to the original activation as
possible. Therefore, changing the meaning by as little as possible.

## Constraints

We define the constraint set based on the barrier function defined in
the ODESteer paper, namely it is the log-density ratio between positive
and negative activations. $$\begin{align*}
    h(a) &= w'^{\top}\phi(a) + b' + \log\left(\frac{N_-}{N_+}\right)
\end{align*}$$ where $\phi(a): \mathbb{R}^d \rightarrow \mathbb{R}^D$ is
a nonlinear feature map and $w'$ and $b'$ represent the learned weights
and biases from the logistic regression used to estimate the density
ratio. Our constraint then is as follows: $$\begin{align*}
    h(a) \geq \epsilon
\end{align*}$$ where $\epsilon$ is the desired distance from the
boundary of the barrier function, which can be thought of as the
probability above 0.5 that the activation is positive. Our perturbed
optimization problem then becomes for each step $\eta$:

$$\begin{aligned}
\min_{a \in \mathbb{R}^d} \quad & \frac{\eta}{2}||a - a_0||_2^2 - \log\left( h(a) - \epsilon\right)\\
\end{aligned}$$


The gradient of this function is: $$\begin{align*}
    \nabla f_\eta(a) = \eta (a - a_0) - \frac{1}{h(a)-\epsilon} J_\phi (a)^\top w'
\end{align*}$$ where $J_\phi (a)$ is the Jacobian of $\phi$ with respect
to $a$. The Hessian of this function is: $$\begin{align*}
    \nabla^2 f_\eta (a) &=2\eta + \frac{1}{(h(a)-\epsilon)^2}\nabla h(a)\nabla h(a)^\top - \frac{1}{h(a) -\epsilon}\nabla^2 h(a)\\
    &=2\eta I + \frac{1}{(h(a)-\epsilon)^2}(J_\phi (a)^\top w')(w'^\top J_\phi (a)) - \frac{1}{h(a) -\epsilon}\nabla^2 h(a)
\end{align*}$$ where $I$ is the identity matrix and if we take $\phi$ to
be a quadratic polynomial count sketch as is done in ODESteer then
$\nabla^2 h(a)$ the hessian of $\phi(a)$ is constant.

We can even expand this definition further if we would like to consider
multiple safety regions at once by defining a unique log-density ratio
for each region and then stacking these constraints. This enables us to
guarantee that the result will be in the intersection of several safety
regions at once.

## Calculating the Hessian

Due to the computational complexity and memory requirements, calculating
and inverting the full hessian $\nabla^2 f_\eta (a)$ is fairly
infeasible. Therefore in order to implement the interior point method we
will have to find a way to calculate or approximate the Newton step
without calculating the full hessian. We can do this using the Conjugate
Gradient method, provided we have access to a function that can
calculate the matrix vector product between the hessian of our function
and a given vector. We simply set up the system of equations and use CG
to approximate the result: $$\begin{align*}
    \nabla^2 f_\eta (a)n_\eta(a) &= \nabla f_\eta (a)\\
    n_\eta(a) &= \left(\nabla^2 f_\eta (a)\right)^{-1} \nabla f_\eta (a)
\end{align*}$$ This works in part because the CG algorithm does not
require the full matrix it only needs to query it using matrix vector
products. This can be implemented efficiently in PyTorch.

## Practical Concerns

### Batching

To improve performance when generating many activations at once we can
batch them by simply stacking them and summing the results of the
objective function for each one. This allows us to solve each
optimization problem simultaneously.

### Stopping Criteria

To decide when to stop optimizing, we simply check the amount of change
between the previous iterate and the current one to see if we have
converged to the optimal solution. We can control the amount of change
that we require between iterates using a hyperparameter. In case of the
algorithm not converging in a reasonable time we also set a max number
of iterates.

### Ensuring Convergence to the Central Path

To ensure convergence to the central path for each value of $\eta_t$ we
actually take all steps $n_{\eta_t}(a_t)$ an inner iteration/loop that
takes multiple steps without changing the value of $\eta_t$ and then we
only update $\eta_t$ as described above once the inner optimization has
converged. This loop has stopping criteria similar to the one described
above.

### Remaining in the Feasible Region

To ensure we remain in the feasible region, we do a back tracking line
search before taking the step $n_{\eta_t}(a_t)$. That is, we check that
after the step the resulting activation $a_{t+1}$ is still in the
feasible region, and if not we can scale $n_{\eta_t}(a_t)$ by $\alpha$
and then check again. If the resulting activation is still not in the
feasible region, we can reduce $\alpha$ and try again. We repeat this
process until we find a step size that remains in the feasible region.

### Other Optimizations

Other small optimizations can be applied for speed; first, we can check
if an activation begins feasible and if so it does not need to be
steered. Secondly, when beginning the steering process, we can linearly
interpolate between the known feasible activation and the initial
activation until we are directly on the border of the feasible region.
This provides a warm start that should be close to the optimal solution.

# Experiments

## Phase 0 Experimentation - ODESteer Execution

The first goal is to get an understanding of the process of steering and
how the ODESteer paper implemented their work to use that as a baseline
for our own experimentation. To that end, before beginning work on
initial testing, an end to end execution of ODESteer should be
performed. The goal of this phase will be to produce a jupyter notebook
that can be run end to end and executes the ODESteer method.

## Phase 1 Experimentation - Small Scale Testing

### Model & Dataset Selection

For initial testing and experimenting it is desirable to remain at a
relatively small scale, then after the initial concept has been refined
we can scale to larger datasets and models. We can use the Falcon3-1B
model, this is a pruned version of the Falcon3-7B model used in the
ODESteer paper with 6 billion less parameters [^4]. This should
make it significantly less computationally expensive to run and a good
testing option before moving to a bigger model. The first datasets we
can test on at this small scale is the \"Jigsaw Unintended Bias in
Toxicity Classification\" kaggle dataset and allenai's
RealToxicityPrompts. The first can be used as a training set by
inputting examples of the toxic and non-toxic texts into model and
extracting the activations. We can then use those activations to define
the log ratio function. The second dataset can serve as a testing set by
inputting the prompts and using OpenAI's moderation API to measure the
toxicity of the resulting generation from the various steering methods.

### Multi-Objective Steering

Another hypothesis we will test during this phase is that when using
traditional one step steering to satisfy one requirement we believe it
will degrade the ability to satisfy another. For example, if the
positive traits we are looking for are positivity and truthfulness,
steering towards positivity will degrade the performance on the
truthfulness metric and vice versa. As stated above we believe interior
point steering can avoid this issue.

To test this we can try out three methods, one being traditional one
step steering where we collect activations associated with both of the
qualities together and their contrastive pairs. The second can be a form
of two step steering where we first apply steering for one requirement
and subsequently the other. The final will be our method where we apply
each requirement as a constraint. We can then compare these, along with
a control to see which performs the best.

### Success Metrics

For the toxicity dataset we will use OpenAI's moderation API which
provides a score from zero to one in several categories, such as
harassment, hate, and other toxic/negative content. This score
represents the probability that content in that category is present in
the text. We can add all the scores from the various categories to get
an overall \"toxicity score\" that we will be trying to reduce.
Simultaneously, we want to confirm that we are not overly degrading the
legibility of the output, for this we can use the well established
perplexity metric which we will try to not raise too much from the
control.

## Phase 2 Experimentation - Large Scale Testing

After confirming our method works in small scale testing we should then
scale up and test using more computational resources. This will involve
testing all other popular steering methods on several different datasets
to gather enough evidence that our method performs better than others.

### Hyperparameter Tuning

There are many tuneable hyper-parameters in our method including the
steering layer, $\eta_0$, $\delta$, and $\epsilon$. For each of these,
we should perform a grid search or some other method of hyperparameter
tuning so we will be able to provide reasonable values for each in the
final work. For some of them it may be possible to derive an optimal
value for them mathematically and in that case we should do so.

## References
[^1] Ben Cottier et al. The rising costs of training frontier AI models. 2025. arXiv: 2405.21015  url: https://arxiv.org/abs/2405.21015.


[^2] Kenneth Li et al. “Inference-Time Intervention: Eliciting Truthful Answers from a Language Model”. In: Advances in Neural Information Process-ing Systems. Ed. by A. Oh et al. Vol. 36. Curran Associates, Inc., 2023, pp 41451–41530. url: https://proceedings.neurips.cc/paper_files/paper/2023/file/81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf


[^3] Nina Panickssery et al. Steering Llama 2 via Contrastive Activation Addition. 2024. arXiv: 2312.06681 url: https://arxiv.org/abs/2312.06681


[^4] Falcon-LLM Team. The Falcon 3 Family of Open Models. Dec. 2024. url:https://huggingface.co/blog/falcon3


[^5] Nisheeth K. Vishnoi. “An Interior Point Method for Linear Programming”.In: Algorithms for Convex Optimization. Cambridge University Press, 2021,pp. 185–214.

[^6] Hongjue Zhao et al. ODESteer: A Unified ODE-Based Steering Frameworkfor LLM Alignment. 2026. arXiv: 2602.17560 url: https://arxiv.org/abs/2602.17560.


[^7] Andy Zou et al. Representation Engineering: A Top-Down Approach to AI Transparency. 2023. arXiv: 2310.01405 